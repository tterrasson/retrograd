use super::*;

// FFI safety contract for this module: scalar out-parameters and temporary
// byte buffers remain live for each synchronous runtime call, and buffer
// lengths are supplied alongside their pointers.

pub(crate) fn kernel_run_info(info: &ffi::RetroKernelRunInfo) -> KernelRunInfo {
    let variant = {
        let bytes: Vec<u8> = info
            .variant
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    KernelRunInfo {
        requested: KernelImpl::from_id(info.requested_impl).unwrap_or(KernelImpl::Auto),
        executed: KernelImpl::from_id(info.executed_impl).unwrap_or(KernelImpl::Native),
        reject: KernelReject::from_id(info.reject_reason),
        variant,
    }
}

/// Snapshot of the process-wide RIR dispatch counters.
///
/// Monotonic since process start and shared by every backend, so a coverage
/// measurement has to diff two snapshots around the run it wants to describe.
pub fn rir_counters() -> Result<RirCounters> {
    let mut raw = ffi::RetroRirCounters::default();
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    if unsafe { ffi::retro_rir_counters_get(&mut raw) } != 0 {
        return Err(runtime_error());
    }
    Ok(RirCounters {
        mode: raw.mode,
        ops_seen: raw.ops_seen,
        rir_eligible: raw.rir_eligible,
        rir_dispatched: raw.rir_dispatched,
        native_dispatched: raw.native_dispatched,
        fallback_contract: raw.fallback_contract,
        fallback_feature: raw.fallback_feature,
        fallback_pipeline: raw.fallback_pipeline,
        reject_by_reason: raw.reject_by_reason,
    })
}

/// Sets the RIR policy for this process. Must be called before any backend
/// context exists - in practice, before the first trainer or probe.
///
/// Asking for the policy already in force always succeeds. Asking for a
/// different one after it has been read is an error, never a silent no-op: the
/// pipelines were built against the earlier mode, so a mode that disagrees with
/// them is exactly how `KernelRunInfo::executed` would start to lie.
pub fn set_rir_runtime_policy(mode: RirMode) -> Result<()> {
    let config = ffi::RetroRuntimeConfig {
        rir_mode: mode.id(),
        ..Default::default()
    };
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    if unsafe { ffi::retro_runtime_config_apply(&config) } != 0 {
        return Err(runtime_error());
    }
    Ok(())
}

/// The RIR policy in force, and whether it can still be changed. The mode is
/// fixed before the first backend context exists, so
/// once `latched` is true a different policy can only be had in a new process.
///
/// Reading it is what latches it, hence `latched` describes the state *before*
/// this call.
pub fn rir_runtime_policy() -> Result<(RirMode, bool)> {
    let mut raw = ffi::RetroRuntimeConfig::default();
    let mut latched = false;
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    if unsafe { ffi::retro_runtime_config_effective(&mut raw, &mut latched) } != 0 {
        return Err(runtime_error());
    }
    Ok((RirMode::from_i32(raw.rir_mode), latched))
}

/// One line per compiled RIR variant
/// (`variant\tkernel\tggml_op\tvariant_id\tbackend\tpriority`) and one per
/// `(ggml_op, backend)` policy row, whose first field is the policy itself:
/// `native-only`, `observe-only` (registered and measured, never encoded) or
/// `prefer`, then `\tggml_op\tbackend`. A `variant` line alone never means the
/// variant runs - the policy line is what says so.
///
/// Followed by the live counter breakdown, one row per site and per variant:
///
/// ```text
/// site\tggml_op\tbackend\tseen=N\teligible=N\trir=N\tnative=N
/// site-variant\tggml_op\tbackend\tvariant_id\tN
/// site-reject\tggml_op\tbackend\treason\tN
/// site-domain\tggml_op\tbackend\treason
/// site-retired\tggml_op\tbackend
/// ```
///
/// `site-domain` is a part of the op's ggml domain the registry publishes as
/// out of scope; a `site-reject` with no matching `site-domain` is a node the
/// kernel claimed and did not serve. `site-retired` says this pair has **no
/// native kernel left** - the strongest thing the report can say about a
/// promotion, and the one thing `native=0` cannot, since a lucky graph
/// produces that on a pair whose native is still there.
pub fn rir_variant_report() -> Result<String> {
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    read_string(|buffer, n_buffer, out| unsafe {
        ffi::retro_rir_variant_report(buffer, n_buffer, out)
    })
}

/// The census of the ggml graphs computed so far, one row per `(ggml_op,
/// backend)` **whether or not RIR covers it** - which is what makes it able to
/// answer "which op should be written next":
///
/// ```text
/// census\tggml_op\tbackend\tnodes\tbytes\telements\tregistered|uncovered
/// census-shape\tggml_op\tbackend\ttype\tne0,ne1,ne2,ne3\tnodes
/// census-shape-overflow\tggml_op\tbackend\tnodes
/// census-pattern\top1>op2[>…]\tbackend\toccurrences\tdispatches\tintermediate_bytes\tregistered|uncovered
/// census-pattern-overflow\twindows
/// ```
///
/// Empty unless `RETRO_RIR_CENSUS=1` is in the environment **before** the first
/// graph is computed. The rows count work, never time: measuring a node's time
/// would need a synchronization per node, whose fixed cost is larger than most
/// nodes and would flatten the very ranking this exists to produce. The time of
/// a shape is what `scripts/test-rir.sh` measures, in isolation.
///
/// The `census-pattern` rows rank an **edge** rather than an op: a fusion does
/// not delete an op, it deletes the boundary between two - one dispatch, and
/// the round trip through memory of a
/// tensor that only existed to cross it. A row is a linear chain of ops where
/// each producer's result is read exactly once in the whole graph, is not a
/// graph output and is not a view, so absorbing it into its consumer is a local
/// rewrite. `dispatches` is what the occurrences cost today and `occurrences`
/// is what they would cost fused; `intermediate_bytes` is what a fused kernel
/// would stop writing, and the traffic it saves is twice that - written, then
/// read back. In-place edges are excluded from that column: the graph builder
/// had already saved them.
pub fn rir_census_report() -> Result<String> {
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    read_string(|buffer, n_buffer, out| unsafe {
        ffi::retro_rir_census_report(buffer, n_buffer, out)
    })
}

/// Runs the registry's variant-selection rule against synthetic tables and
/// returns the bitmask of failing cases - `0` when the rule held everywhere.
///
/// The shipped registry carries one variant per `(op, backend)`, so nothing in
/// it could distinguish "the priority decided" from "the first matching row was
/// taken". This drives the same selection function production goes through, on
/// tables built for the purpose. No device needed.
pub fn rir_selection_selftest() -> u32 {
    // SAFETY: the module contract keeps each scalar or buffer out-parameter live for the synchronous call.
    unsafe { ffi::retro_rir_selection_selftest() }
}
