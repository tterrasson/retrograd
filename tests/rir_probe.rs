//! RIR integration contract at the FFI boundary.
//!
//! Two things are checked here, and neither needs a model:
//!
//! 1. the AOT registry the backends dispatch from is the one the FFI reports,
//!    same variants, same natively-locked ops;
//! 2. `probe_op_ex` reports *which* implementation ran, and refuses to report
//!    RIR when the native kernel is what actually executed.
//!
//! The RIR variant itself can only run when the process was started with
//! `RETRO_RIR_MODE=prefer` (the policy has to be fixed before the backend
//! context exists), so this file splits: without it, the tests assert the clean
//! refusal; with it, they assert an actual RIR dispatch and its parity against
//! the native kernel. `scripts/test-abi.sh` runs the binary both ways.

mod common;

use retrograd::{
    KernelImpl, ProbeInputs, ProbeOp, RirMode, probe_op_ex, rir_counters, rir_runtime_policy,
    rir_selection_selftest, rir_variant_report, set_rir_runtime_policy,
};

/// Deterministic pseudo-random f32s in [-range, range] (no rand dependency).
fn pseudo_random(n: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
            let unit = (bits as f32) / ((1u32 << 24) as f32);
            (unit * 2.0 - 1.0) * range
        })
        .collect()
}

/// Whether RIR may encode in this process. See [`common::rir_encoding_enabled`]
/// for why an unset variable reads as `prefer`.
fn prefer_mode() -> bool {
    common::rir_encoding_enabled()
}

fn require_mode() -> bool {
    matches!(std::env::var("RETRO_RIR_MODE").as_deref(), Ok("require"))
}

/// Whether this process was started with the test-only contract injection that
/// makes L2_NORM_BACK ineligible. There is no shape the probe API can build
/// that the variant rejects - every remaining rejection needs a tensor too
/// large to allocate or a device offset the caller cannot choose - so the
/// rejecting path of `require` is reached this way.
fn reject_injected() -> bool {
    std::env::var("RETRO_RIR_TEST_REJECT")
        .map(|v| v.split(',').any(|op| op == "L2_NORM_BACK"))
        .unwrap_or(false)
}

/// Whether this process promotes the CUMSUM variant from `observe_generated`
/// to `prefer_generated`. The registry keeps it registered-but-not-encoded
/// until a benchmark says otherwise; this override is how the differential
/// matrix measures it in the meantime.
fn cumsum_promoted() -> bool {
    std::env::var("RETRO_RIR_TEST_PREFER")
        .map(|v| v.split(',').any(|op| op == "CUMSUM"))
        .unwrap_or(false)
}

/// One 2D L2_NORM_BACK case: [ne0, nrows] with `eps`.
struct Case {
    ne0: i64,
    nrows: i64,
    eps: f32,
}

const CASES: &[Case] = &[
    // In-contract: 2D, F32, contiguous, one row and many rows, plus a width
    // that is not a multiple of the 32-lane subgroup the variant reduces over.
    Case {
        ne0: 64,
        nrows: 1,
        eps: 1e-6,
    },
    Case {
        ne0: 65,
        nrows: 600,
        eps: 1e-6,
    },
    // eps above the norm: exercises the branch where the kernel must emit the
    // unscaled gradient instead of the projected one.
    Case {
        ne0: 128,
        nrows: 7,
        eps: 1e3,
    },
];

fn run_case(
    case: &Case,
    use_gpu: bool,
    implementation: KernelImpl,
) -> (Vec<f32>, String, KernelImpl) {
    let n = (case.ne0 * case.nrows) as usize;
    let ne = [case.ne0, case.nrows, 1, 1];
    let dz = pseudo_random(n, 0x51ED_2701, 1.0);
    let x = pseudo_random(n, 0x9E37_79B9, 1.0);
    let (out, info) = probe_op_ex(
        ProbeOp::L2NormBack,
        use_gpu,
        ProbeInputs::pair(ne, &dz, ne, &x),
        [case.eps, 0.0],
        n,
        implementation,
    )
    .unwrap_or_else(|e| {
        panic!(
            "probe {implementation:?} failed on {}x{}: {e}",
            case.ne0, case.nrows
        )
    });
    assert_eq!(
        info.requested, implementation,
        "the run info must echo the requested implementation"
    );
    (out, info.variant, info.executed)
}

/// The RIR counters are **process-global**, and `cargo test` runs this file on
/// several threads. A test that measures a *delta* around one probe is only
/// meaningful while nothing else dispatches.
///
/// That is not hypothetical: `require_fails_the_graph_instead_of_falling_back`
/// asserts that a refused graph moves no counter, and its neighbour
/// `require_leaves_a_native_only_backend_alone` runs a CPU probe of the same op,
/// which moves `ops_seen`, `rir_eligible` and `native_dispatched` by exactly
/// one each. The two are the only tests the `require_` filter of
/// `scripts/test-abi.sh` selects, so they always met.
///
/// Every dispatching test takes this lock, not only the measuring ones:
/// exclusion has to be mutual to be exclusion.
fn serialized_dispatch() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The registry the FFI reports must be the one the backends were built from.
/// A variant added on one side and not the other shows up here as a missing
/// line rather than as a silent native fallback.
#[test]
fn variant_report_matches_the_registry() {
    let report = rir_variant_report().expect("rir_variant_report");
    let variants: Vec<&str> = report
        .lines()
        .filter(|line| line.starts_with("variant\t"))
        .collect();
    assert!(
        !variants.is_empty(),
        "no RIR variant compiled in; the registry should carry at least l2_norm_back:\n{report}"
    );
    for line in &variants {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 6, "malformed variant line: {line}");
        assert!(
            fields[2].starts_with("GGML_OP_"),
            "a production variant must name a real ggml op: {line}"
        );
        assert!(!fields[3].is_empty(), "variant_id must be stable: {line}");
    }
    let has_variant = |op: &str, backend: &str| {
        report.lines().any(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            f.first() == Some(&"variant") && f.get(2) == Some(&op) && f.get(4) == Some(&backend)
        })
    };
    let policy = |op: &str, backend: &str| -> Option<&str> {
        report
            .lines()
            .find(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                f.get(1) == Some(&op)
                    && f.get(2) == Some(&backend)
                    && matches!(
                        f.first(),
                        Some(&("native-only" | "observe-only" | "prefer" | "prefer-only"))
                    )
            })
            .and_then(|l| l.split('\t').next())
    };

    // L2_NORM_BACK is the pair whose native kernel is gone on **all three** GPU
    // backends: Metal and Vulkan first, then CUDA.
    //
    // A hand-pinned answer for one backend outlives the decision it recorded,
    // and only breaks once that backend actually ships. So the rung is asserted
    // the same way for every backend that carries the variant, and a backend
    // absent from this build carries none - which is not a failure.
    for backend in ["metal", "vulkan", "cuda"] {
        if !has_variant("GGML_OP_L2_NORM_BACK", backend) {
            continue;
        }
        // `prefer-only`, not `prefer`: the native kernel of this op is gone, and
        // the report has a rung of its own for that. Asserting the weaker
        // spelling would let the removal be undone without a single test
        // noticing - the pair would still be "preferred", which is the whole
        // difference the rung exists to carry.
        assert_eq!(
            policy("GGML_OP_L2_NORM_BACK", backend),
            Some("prefer-only"),
            "L2_NORM_BACK must be the *only* implementation on {backend}:\n{report}"
        );
    }

    // Two distinct ops are registered and selected by the registry, and the two
    // are at different stages: L2_NORM_BACK replaces its native kernel above,
    // CUMSUM is registered for measurement only. Both facts are read from the
    // same report, so neither can be inferred from the other.
    //
    // Metal and Vulkan only: CUMSUM has no CUDA variant, and its CUDA row is
    // `native-only` for that reason rather than as a policy about the scan.
    for backend in ["metal", "vulkan"] {
        // A backend absent from this build has neither, and that is not a
        // failure: what must never happen is a variant with no policy row.
        if !has_variant("GGML_OP_L2_NORM_BACK", backend) && !has_variant("GGML_OP_CUMSUM", backend)
        {
            continue;
        }
        assert!(
            has_variant("GGML_OP_L2_NORM_BACK", backend) && has_variant("GGML_OP_CUMSUM", backend),
            "both integrated ops must carry a {backend} variant:\n{report}"
        );
        assert_eq!(
            policy("GGML_OP_CUMSUM", backend),
            Some("observe-only"),
            "CUMSUM is registered for measurement only until a benchmark promotes \
             it on {backend}:\n{report}"
        );
    }
}

/// Priority - not the order the rows happen to sit in - decides which variant a
/// site gets, and `rir_op_policies` can veto a row that would otherwise win.
///
/// The registry ships one variant per `(op, backend)`, so this cannot be
/// asserted on the real table: with a single candidate, "highest priority" and
/// "first match" are the same answer. The self-test drives the production
/// selection function over tables built for the purpose, which is what makes
/// the published `priority` load-bearing rather than descriptive.
#[test]
fn priority_and_policy_decide_the_selected_variant() {
    const CASES: [(u32, &str); 10] = [
        (1 << 0, "highest priority first in the table must win"),
        (1 << 1, "highest priority last in the table must still win"),
        (1 << 2, "a NATIVE_ONLY policy must veto the variant"),
        (
            1 << 3,
            "a pair absent from the policy table must select nothing",
        ),
        (1 << 4, "equal priorities must fall back to table order"),
        (1 << 5, "a row for another backend must never be selected"),
        (
            1 << 6,
            "an OBSERVE_GENERATED policy must still select, by priority",
        ),
        (
            1 << 14,
            "an identity fold on a contiguous operand must select the vec4 row",
        ),
        (
            1 << 15,
            "a real repeat on the vectorized axis must fall to the scalar row",
        ),
        (
            1 << 16,
            "a permuted folded operand must fall to the scalar row: the stride \
             check on the vectorized axis never looks at it",
        ),
    ];
    let failures = rir_selection_selftest();
    let failed: Vec<&str> = CASES
        .iter()
        .filter(|(bit, _)| failures & bit != 0)
        .map(|(_, what)| *what)
        .collect();
    assert!(
        failed.is_empty(),
        "selection rule broken (mask {failures:#b}): {failed:?}"
    );
}

/// The counters are readable and self-consistent whatever the mode is: they are
/// the only way a coverage measurement can tell a RIR dispatch from a fallback.
#[test]
fn counters_are_readable_and_consistent() {
    let _serialized = serialized_dispatch();
    let c = rir_counters().expect("rir_counters");
    assert!(
        (0..=3).contains(&c.mode),
        "unknown RIR mode reported: {}",
        c.mode
    );
    assert!(
        c.rir_dispatched <= c.rir_eligible,
        "more dispatches than eligible ops: {c:?}"
    );
    assert!(
        c.rir_eligible <= c.ops_seen,
        "more eligible ops than seen: {c:?}"
    );
    assert_eq!(
        c.mode >= 2,
        prefer_mode(),
        "the reported mode must be the configured one: {c:?}"
    );
}

/// The policy is a versioned runtime option, not only an environment
/// variable, and what it reports must be what is in force.
#[test]
fn runtime_policy_reports_the_effective_mode() {
    let (mode, _) = rir_runtime_policy().expect("rir_runtime_policy");
    assert_eq!(
        matches!(mode, RirMode::Prefer | RirMode::Require),
        prefer_mode(),
        "the reported policy must be the configured one, got {mode:?}"
    );
    // Reading it latches it: a second read must say so, which is what makes
    // "the mode cannot change under you" observable rather than documented.
    let (again, latched) = rir_runtime_policy().expect("rir_runtime_policy");
    assert_eq!(again, mode, "the policy changed between two reads");
    assert!(latched, "reading the policy must latch it");
}

/// Once latched, a trainer asking for a *different* policy must fail rather than
/// get a mode the pipelines were not built for. Nothing is loaded here - the
/// refusal happens before the model path is even looked at, which is the point:
/// it is a policy check, not a load failure.
#[test]
fn a_conflicting_policy_is_refused() {
    // Latch it.
    let (mode, _) = rir_runtime_policy().expect("rir_runtime_policy");
    let conflicting = if mode == RirMode::Off {
        RirMode::Prefer
    } else {
        RirMode::Off
    };
    let err = set_rir_runtime_policy(conflicting)
        .expect_err("a conflicting policy must be refused, not silently ignored");
    let text = err.to_string();
    assert!(
        text.contains("RIR policy"),
        "the refusal must name the policy so the cause is actionable: {text}"
    );
    // Asking for the policy already in force is not a conflict.
    set_rir_runtime_policy(mode).expect("re-asserting the current policy must succeed");
    // And the policy really did not move.
    let (after, _) = rir_runtime_policy().expect("rir_runtime_policy");
    assert_eq!(after, mode, "a refused policy must change nothing");
}

/// Forcing the native kernel must work in every mode and must never claim RIR.
///
/// On the **CPU**, where L2_NORM_BACK is native-only by policy and its native
/// kernel is the only one there has ever been. The GPU half of this claim is
/// the test below: there, the native kernel is gone.
#[test]
fn native_is_forced_and_reported() {
    let _serialized = serialized_dispatch();
    for case in CASES {
        let (_, variant, executed) = run_case(case, false, KernelImpl::Native);
        assert_eq!(executed, KernelImpl::Native);
        assert!(variant.is_empty(), "native ran but a variant was reported");
    }
}

/// On a GPU, forcing the native kernel of a pair whose native has been retired
/// must be a **clean error**.
///
/// It is the mirror of `rir_request_without_the_mode_is_a_clean_error`, and it
/// guards the same class of vacuous green. Answering the request on the CPU
/// would let a parity test go on comparing "GPU native" against "GPU RIR" long
/// after one of the two stopped existing, and every assertion in it would keep
/// passing. Honouring it on the GPU cannot work at all: force-native lowers the
/// dispatch mode under a graph whose split `ggml_rir_supports_op` already
/// decided, so the node would reach a site with nothing to encode.
///
/// This is also the only *external* proof that the removal happened. Everything
/// else - a green matrix, 100 % coverage - reads the same whether the native
/// kernel is gone or merely unused.
#[test]
fn forcing_a_retired_native_on_the_gpu_is_a_clean_error() {
    let _serialized = serialized_dispatch();
    if !common::gpu_device_present() {
        eprintln!("skipped: no usable GPU backend in this build");
        return;
    }
    let case = &CASES[0];
    let n = (case.ne0 * case.nrows) as usize;
    let ne = [case.ne0, case.nrows, 1, 1];
    let data = pseudo_random(n, 1, 1.0);
    let err = probe_op_ex(
        ProbeOp::L2NormBack,
        true,
        ProbeInputs::pair(ne, &data, ne, &data),
        [case.eps, 0.0],
        n,
        KernelImpl::Native,
    )
    .expect_err("forcing a retired native must fail, not answer elsewhere");
    let text = err.to_string();
    assert!(
        text.contains("retired"),
        "the error must say the native kernel is gone, so the cause is actionable: {text}"
    );
}

/// Off `prefer`, requesting RIR must fail with a readable error rather than
/// running the native kernel and reporting success. That refusal is what keeps
/// a "RIR is green" claim from being vacuous.
#[test]
fn rir_request_without_the_mode_is_a_clean_error() {
    let _serialized = serialized_dispatch();
    if prefer_mode() {
        eprintln!("skipped: RETRO_RIR_MODE enables RIR in this process");
        return;
    }
    let case = &CASES[0];
    let n = (case.ne0 * case.nrows) as usize;
    let ne = [case.ne0, case.nrows, 1, 1];
    let data = pseudo_random(n, 1, 1.0);
    let err = probe_op_ex(
        ProbeOp::L2NormBack,
        common::gpu_device_present(),
        ProbeInputs::pair(ne, &data, ne, &data),
        [case.eps, 0.0],
        n,
        KernelImpl::Rir,
    )
    .expect_err("requesting RIR without the mode must fail, not fall back");
    let text = err.to_string();
    assert!(
        text.contains("RIR"),
        "the error must name RIR so the cause is actionable: {text}"
    );
}

/// Under `prefer`, the GPU variant must actually run - asserted through
/// `executed_impl`, never inferred from timing - and must match the CPU
/// reference within the published tolerance (the RIR shader reduces in F32
/// while the CPU kernel accumulates in double, so parity is relative, never
/// bit-exact).
///
/// There is no **native GPU kernel** to compare against: L2_NORM_BACK declares
/// no restriction on its ggml domain, so its native kernel was retired, and the
/// independent reference is the CPU oracle here plus the Loop IR device parity
/// in `rir-runtime`. Asking for the native path is a
/// clean error, and the test below asserts *that* rather than silently
/// measuring RIR against itself.
#[test]
fn rir_variant_runs_and_matches_native() {
    let _serialized = serialized_dispatch();
    if !prefer_mode() {
        eprintln!("skipped: set RETRO_RIR_MODE=prefer to exercise the RIR variant");
        return;
    }
    if !common::gpu_device_present() {
        eprintln!("skipped: no usable GPU backend in this build");
        return;
    }

    let before = rir_counters().expect("rir_counters");
    for case in CASES {
        let (rir, variant, executed) = run_case(case, true, KernelImpl::Rir);
        assert_eq!(
            executed,
            KernelImpl::Rir,
            "{}x{} reported {executed:?} instead of RIR",
            case.ne0,
            case.nrows
        );
        assert!(!variant.is_empty(), "a RIR dispatch must name its variant");

        let (cpu, _, _) = run_case(case, false, KernelImpl::Native);

        // Relative to the CPU double-accumulated reference, scaled by the
        // reduction length: a longer row means more F32 rounding in the shader.
        let tol = 1e-5 * (case.ne0 as f32).sqrt();
        for (i, (r, c)) in rir.iter().zip(&cpu).enumerate() {
            let scale = c.abs().max(1.0);
            assert!(
                (r - c).abs() <= tol * scale,
                "RIR vs CPU mismatch at {i} on {}x{} ({variant}): rir={r} cpu={c}",
                case.ne0,
                case.nrows
            );
        }
    }
    let after = rir_counters().expect("rir_counters");
    assert!(
        after.rir_dispatched > before.rir_dispatched,
        "executed_impl claimed RIR but the dispatch counter did not move: \
         before={before:?} after={after:?}"
    );

    // The same dispatches, attributed to (op, backend, variant). With one
    // integrated op these rows restate the aggregate; the point is that they
    // exist and are keyed, because with a second op the aggregate alone can no
    // longer say which op is uncovered.
    let report = rir_variant_report().expect("rir_variant_report");
    let site = report
        .lines()
        .find(|l| l.starts_with("site\tGGML_OP_L2_NORM_BACK\t"))
        .unwrap_or_else(|| panic!("no site row for the op that just dispatched:\n{report}"));
    assert!(
        site.contains("\trir=") && !site.contains("\trir=0\t"),
        "the site row must credit the dispatches it saw: {site}"
    );
    let variant_row = report
        .lines()
        .find(|l| l.starts_with("site-variant\tGGML_OP_L2_NORM_BACK\t"))
        .unwrap_or_else(|| panic!("no per-variant row:\n{report}"));
    let fields: Vec<&str> = variant_row.split('\t').collect();
    assert_eq!(fields.len(), 5, "malformed site-variant row: {variant_row}");
    assert_ne!(
        fields[3], "<overflow>",
        "the variant axis overflowed its per-site slots, so the breakdown is lossy"
    );
    assert!(
        report.lines().any(|l| l
            == format!(
                "variant\tl2_norm_back\tGGML_OP_L2_NORM_BACK\t{}\t{}\t80",
                fields[3], fields[2]
            )),
        "the variant a site credited must be a registry row:\n{report}"
    );
}

/// A rank-3 tensor runs on the variant's own outer axes - the kernel carries
/// `i1`, `i2` and `i3` with one `nb[]` per argument, so nothing has to fold.
/// This is the shape the Qwen3.5 graph produces (`ne=[128,16,16,1]`).
/// The probe builds contiguous buffers, so the
/// *gapped* variant of this shape lives in `tests/rir_graph_coverage.rs` and in
/// the fork's `test-backend-ops` case, not here.
#[test]
fn a_rank3_shape_runs_on_the_variants_outer_axes() {
    let _serialized = serialized_dispatch();
    if !prefer_mode() || !common::gpu_device_present() {
        eprintln!("skipped: needs RETRO_RIR_MODE=prefer and a GPU backend");
        return;
    }
    // The real graph shape, plus a smaller one whose row count is not a
    // multiple of anything convenient.
    for ne in [[128_i64, 16, 16, 1], [32, 3, 5, 1]] {
        let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
        let dz = pseudo_random(n, 7, 1.0);
        let x = pseudo_random(n, 11, 1.0);
        let (rir, info) = probe_op_ex(
            ProbeOp::L2NormBack,
            true,
            ProbeInputs::pair(ne, &dz, ne, &x),
            [1e-6, 0.0],
            n,
            KernelImpl::Rir,
        )
        .unwrap_or_else(|e| panic!("rank-3 {ne:?} must run on the variant: {e}"));
        assert_eq!(info.executed, KernelImpl::Rir);

        let (cpu, _) = probe_op_ex(
            ProbeOp::L2NormBack,
            false,
            ProbeInputs::pair(ne, &dz, ne, &x),
            [1e-6, 0.0],
            n,
            KernelImpl::Native,
        )
        .expect("CPU reference");
        let tol = 1e-5 * (ne[0] as f32).sqrt();
        for (i, (r, c)) in rir.iter().zip(&cpu).enumerate() {
            assert!(
                (r - c).abs() <= tol * c.abs().max(1.0),
                "rank-3 {ne:?} mismatch at {i}: rir={r} cpu={c}"
            );
        }
    }
}

/// `require` must fail the graph rather than run the native kernel: the
/// preflight evaluates every targeted node before anything is encoded, so an
/// ineligible one is an error, not a fallback.
///
/// The counters are the proof that this happened *before* dispatch: neither
/// `native_dispatched` nor `ops_seen` may move, because the dispatch site was
/// never reached. A fallback would move both.
#[test]
fn require_fails_the_graph_instead_of_falling_back() {
    let _serialized = serialized_dispatch();
    if !require_mode() || !reject_injected() {
        eprintln!("skipped: needs RETRO_RIR_MODE=require and RETRO_RIR_TEST_REJECT=L2_NORM_BACK");
        return;
    }
    if !common::gpu_device_present() {
        eprintln!("skipped: no usable GPU backend in this build");
        return;
    }
    let case = &CASES[0];
    let n = (case.ne0 * case.nrows) as usize;
    let ne = [case.ne0, case.nrows, 1, 1];
    let data = pseudo_random(n, 3, 1.0);

    // `Auto` on purpose: nothing in the request asks for RIR, so what fails is
    // the graph policy itself and not the probe's own strictness about a RIR
    // request. `Rir` must fail too, and both must fail the same way.
    for implementation in [KernelImpl::Auto, KernelImpl::Rir] {
        let before = rir_counters().expect("rir_counters");
        let err = probe_op_ex(
            ProbeOp::L2NormBack,
            true,
            ProbeInputs::pair(ne, &data, ne, &data),
            [case.eps, 0.0],
            n,
            implementation,
        )
        .expect_err("under require an ineligible targeted node must fail the graph");
        let text = err.to_string();
        assert!(
            text.contains("require") && text.contains("GGML_OP_L2_NORM_BACK"),
            "the error must name the policy and the op it refused: {text}"
        );
        let after = rir_counters().expect("rir_counters");
        assert_eq!(
            after.native_dispatched, before.native_dispatched,
            "require fell back to the native kernel: before={before:?} after={after:?}"
        );
        assert_eq!(
            after.rir_dispatched, before.rir_dispatched,
            "nothing may be dispatched after a preflight refusal: {after:?}"
        );
        assert_eq!(
            after.ops_seen, before.ops_seen,
            "the refusal must happen before the dispatch site, not at it: \
             before={before:?} after={after:?}"
        );
    }
}

/// `require` constrains the nodes the registry *targets*, and only those. On
/// the CPU backend L2_NORM_BACK is native-only by policy, so the same graph
/// that the GPU refuses must still run - otherwise `require` would mean "no
/// backend may run this op", which is not what it promises.
#[test]
fn require_leaves_a_native_only_backend_alone() {
    let _serialized = serialized_dispatch();
    if !require_mode() || !reject_injected() {
        eprintln!("skipped: needs RETRO_RIR_MODE=require and RETRO_RIR_TEST_REJECT=L2_NORM_BACK");
        return;
    }
    let case = &CASES[0];
    let n = (case.ne0 * case.nrows) as usize;
    let ne = [case.ne0, case.nrows, 1, 1];
    let data = pseudo_random(n, 5, 1.0);
    let (out, info) = probe_op_ex(
        ProbeOp::L2NormBack,
        false,
        ProbeInputs::pair(ne, &data, ne, &data),
        [case.eps, 0.0],
        n,
        KernelImpl::Auto,
    )
    .expect("CPU is native-only for this op, so require must not refuse it");
    assert_eq!(info.executed, KernelImpl::Native);
    assert_eq!(out.len(), n);
}

// ---------------------------------------------------------------------------
// GGML_OP_CUMSUM - the second integrated op
//
// Registered on Metal and Vulkan with the `observe_generated` policy: the
// registry selects a variant, the dispatch site evaluates its contract and
// counts the result, and the native blocked scan runs. The RIR scan is
// sequential *within* one invocation while the native kernels are blocked, so
// promotion waits on a benchmark - which is what `RETRO_RIR_TEST_PREFER`
// exists to make measurable.
// ---------------------------------------------------------------------------

/// Shapes the CUMSUM matrix walks: ranks 2 to 4, widths that are not multiples
/// of the 64-wide workgroup, a row count large enough to need several
/// workgroups in x, and a long row so the sequential scan is not trivial.
const CUMSUM_SHAPES: &[[i64; 4]] = &[
    [64, 1, 1, 1],
    [65, 600, 1, 1],
    [1, 4096, 1, 1],
    [128, 16, 16, 1],
    [7, 3, 5, 2],
    [4096, 2, 1, 1],
];

fn run_cumsum(
    ne: [i64; 4],
    use_gpu: bool,
    implementation: KernelImpl,
) -> (Vec<f32>, String, KernelImpl) {
    let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
    let x = pseudo_random(n, 0x0C0F_FEE1 ^ (ne[0] as u64), 1.0);
    let (out, info) = probe_op_ex(
        ProbeOp::Cumsum,
        use_gpu,
        // src1 is ignored by the op, but the probe ABI always carries two inputs.
        ProbeInputs::pair(ne, &x, ne, &x),
        [0.0, 0.0],
        n,
        implementation,
    )
    .unwrap_or_else(|e| panic!("cumsum probe {implementation:?} failed on {ne:?}: {e}"));
    assert_eq!(info.requested, implementation);
    (out, info.variant, info.executed)
}

/// The reference: an inclusive prefix sum per row, accumulated in f64 so the
/// comparison is against the mathematics and not against another f32 order.
fn cumsum_reference(x: &[f32], ne: [i64; 4]) -> Vec<f32> {
    // The probe allocates contiguous tensors, so the outer three dimensions are
    // just consecutive rows of `ne0` elements.
    let n_col = ne[0] as usize;
    let n_rows_total = x.len() / n_col.max(1);
    let mut out = vec![0f32; x.len()];
    for r in 0..n_rows_total {
        let mut acc = 0f64;
        for c in 0..n_col {
            acc += x[r * n_col + c] as f64;
            out[r * n_col + c] = acc as f32;
        }
    }
    out
}

/// Under `observe_generated` the site must *measure* CUMSUM without ever
/// encoding it: the node is seen, its contract is found eligible, and the
/// native kernel runs. Asserting all three is what separates "registered and
/// measured" from "registered and quietly skipped".
#[test]
fn cumsum_is_measured_but_not_dispatched() {
    let _serialized = serialized_dispatch();
    if !prefer_mode() || cumsum_promoted() {
        eprintln!("skipped: needs RETRO_RIR_MODE=prefer without RETRO_RIR_TEST_PREFER=CUMSUM");
        return;
    }
    if !common::gpu_device_present() {
        eprintln!("skipped: no usable GPU backend in this build");
        return;
    }

    for &ne in CUMSUM_SHAPES {
        let (out, variant, executed) = run_cumsum(ne, true, KernelImpl::Auto);
        assert_eq!(
            executed,
            KernelImpl::Native,
            "{ne:?}: an observe-only pair must never encode its variant"
        );
        assert!(
            variant.is_empty(),
            "{ne:?}: native ran but a variant was reported"
        );

        // The measurement is only worth anything if it is also correct.
        let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
        let x = pseudo_random(n, 0x0C0F_FEE1 ^ (ne[0] as u64), 1.0);
        let expected = cumsum_reference(&x, ne);
        let tol = 1e-5 * (ne[0] as f32).sqrt();
        for (i, (g, e)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= tol * e.abs().max(1.0),
                "{ne:?}: native cumsum mismatch at {i}: got={g} want={e}"
            );
        }
    }
    // The per-site breakdown is what makes a second op readable at all - and
    // the only thing that *can* be asserted here. The aggregate counters are
    // process-wide, so the L2_NORM_BACK tests sharing this binary move them;
    // only these rows say which op was covered and which was merely observed.
    // That is precisely the ventilation this per-op breakdown exists to
    // put to use.
    let report = rir_variant_report().expect("rir_variant_report");
    let site = report
        .lines()
        .find(|l| l.starts_with("site\tGGML_OP_CUMSUM\t"))
        .unwrap_or_else(|| panic!("no site row for CUMSUM after it ran:\n{report}"));
    assert!(
        site.contains("\trir=0\t"),
        "the CUMSUM site must credit zero RIR dispatches: {site}"
    );
    let eligible = site
        .split('\t')
        .find_map(|f| f.strip_prefix("eligible="))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("malformed site row: {site}"));
    assert!(
        eligible > 0,
        "every shape of the matrix is inside the variant's contract, so the \
         observation must find them eligible: {site}"
    );
}

/// Requesting the RIR variant of an observe-only pair must fail by naming the
/// policy, not run the native kernel and report success.
#[test]
fn a_rir_request_on_an_observe_only_pair_is_refused() {
    let _serialized = serialized_dispatch();
    if !prefer_mode() || cumsum_promoted() || !common::gpu_device_present() {
        eprintln!("skipped: needs a GPU, RETRO_RIR_MODE=prefer and no CUMSUM promotion");
        return;
    }
    let ne = CUMSUM_SHAPES[0];
    let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
    let x = pseudo_random(n, 17, 1.0);
    let err = probe_op_ex(
        ProbeOp::Cumsum,
        true,
        ProbeInputs::pair(ne, &x, ne, &x),
        [0.0, 0.0],
        n,
        KernelImpl::Rir,
    )
    .expect_err("an observe-only pair must refuse a RIR request");
    let text = err.to_string();
    assert!(
        text.contains("observe_generated"),
        "the refusal must name the policy that caused it: {text}"
    );
}

/// `require` constrains only the pairs the registry *targets*. CUMSUM is
/// registered for observation, so an injected contract rejection on it must
/// leave the graph alone - otherwise adding a second op to the registry would
/// silently start failing graphs the first op never touched.
#[test]
fn require_ignores_an_observe_only_pair() {
    let _serialized = serialized_dispatch();
    if !require_mode() || !common::gpu_device_present() {
        eprintln!("skipped: needs RETRO_RIR_MODE=require and a GPU backend");
        return;
    }
    if cumsum_promoted() {
        eprintln!("skipped: CUMSUM is promoted in this process");
        return;
    }
    let ne = CUMSUM_SHAPES[1];
    let (out, _, executed) = run_cumsum(ne, true, KernelImpl::Auto);
    assert_eq!(executed, KernelImpl::Native);
    assert_eq!(out.len(), (ne[0] * ne[1] * ne[2] * ne[3]) as usize);
}

/// With the pair promoted, the variant must actually run on every shape of the
/// matrix and agree with both the native GPU kernel and the f64 reference.
/// This is the differential measurement the promotion decision needs; until it
/// is green *and* a benchmark is non-regressive, the registry keeps the pair
/// on `observe_generated`.
#[test]
fn a_promoted_cumsum_matches_native_and_the_reference() {
    let _serialized = serialized_dispatch();
    if !prefer_mode() || !cumsum_promoted() {
        eprintln!("skipped: needs RETRO_RIR_MODE=prefer and RETRO_RIR_TEST_PREFER=CUMSUM");
        return;
    }
    if !common::gpu_device_present() {
        eprintln!("skipped: no usable GPU backend in this build");
        return;
    }

    let before = rir_counters().expect("rir_counters");
    for &ne in CUMSUM_SHAPES {
        let (rir, variant, executed) = run_cumsum(ne, true, KernelImpl::Rir);
        assert_eq!(executed, KernelImpl::Rir, "{ne:?} did not run the variant");
        assert!(!variant.is_empty(), "a RIR dispatch must name its variant");

        let (native, _, native_executed) = run_cumsum(ne, true, KernelImpl::Native);
        assert_eq!(native_executed, KernelImpl::Native);

        let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
        let x = pseudo_random(n, 0x0C0F_FEE1 ^ (ne[0] as u64), 1.0);
        let expected = cumsum_reference(&x, ne);
        // A prefix sum accumulates the whole row, so the error budget grows
        // with the row length in both implementations.
        let tol = 1e-5 * (ne[0] as f32).sqrt();
        for (i, ((r, nat), e)) in rir.iter().zip(&native).zip(&expected).enumerate() {
            let scale = e.abs().max(1.0);
            assert!(
                (r - e).abs() <= tol * scale,
                "{ne:?} ({variant}): RIR vs reference at {i}: rir={r} want={e}"
            );
            assert!(
                (r - nat).abs() <= tol * scale,
                "{ne:?}: RIR vs native GPU at {i}: rir={r} native={nat}"
            );
        }
    }
    let after = rir_counters().expect("rir_counters");
    assert!(
        after.rir_dispatched > before.rir_dispatched,
        "executed_impl claimed RIR but the dispatch counter did not move: \
         before={before:?} after={after:?}"
    );

    // The dispatches must be attributed to CUMSUM's own site - the property
    // that only becomes testable once a second op exists.
    let report = rir_variant_report().expect("rir_variant_report");
    let site = report
        .lines()
        .find(|l| l.starts_with("site\tGGML_OP_CUMSUM\t"))
        .unwrap_or_else(|| panic!("no CUMSUM site row:\n{report}"));
    assert!(
        !site.contains("\trir=0\t"),
        "the CUMSUM site must credit its own dispatches: {site}"
    );
    let variant_row = report
        .lines()
        .find(|l| l.starts_with("site-variant\tGGML_OP_CUMSUM\t"))
        .unwrap_or_else(|| panic!("no per-variant row for CUMSUM:\n{report}"));
    let fields: Vec<&str> = variant_row.split('\t').collect();
    assert_eq!(fields.len(), 5, "malformed site-variant row: {variant_row}");
    assert!(
        report
            .lines()
            .any(|l| l.starts_with(&format!("variant\tcumsum\tGGML_OP_CUMSUM\t{}\t", fields[3]))),
        "the variant CUMSUM credited must be a registry row:\n{report}"
    );
}
