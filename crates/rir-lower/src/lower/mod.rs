//! Lowering from semantic IR to Loop IR.
//!
//! Supported form: kernels with `K >= 2` logical axes and at most one **inner**
//! reduction or scan axis; all other axes are parallel. Parallel axes map to
//! the schedule's `grid_dims` grid dimensions at the level selected by
//! `ParallelMapping`; any remaining axes become sequential loops. Two
//! strategies are supported:
//!
//! - `Serial` (CPU and GPU without collectives):
//!   ```text
//!   parallel p0... p_{g-1}               (grid)
//!     for p_g... pn                      (remaining parallel axes)
//!       // reductions, one pass per dependency level:
//!       init accs; for col { phase };...; per-row writes;
//!       for col { phase 2: writes that depend on col }
//!       // scans (mutually exclusive with reductions in v1):
//!       init states; for col (scan direction) { input, state, writes }
//!   ```
//! - `SubgroupTree` (GPU, reductions):
//!   ```text
//!   parallel p0... p_{g-1}               (grid: one workgroup per index)
//!     parallel lane: 0..lanes
//!       // one pass per reduction dependency level:
//!       init partial accs; for col = lane.. n step lanes { phase }
//!       lanereduce accs -> reds           (broadcast to every lane)
//!       if lane == 0 { per-row writes }
//!       for col = lane.. n step lanes { phase 2 }
//!   ```
//! - `SubgroupTree` + `ScanStrategy::BlockedLanes` (GPU, scans):
//!   ```text
//!   parallel p0... p_{g-1}               (grid: one workgroup per index)
//!     parallel lane: 0..lanes
//!       init part; forchunk col in lane's chunk { part += input }
//!       lanescan part -> off              (exclusive prefix across lanes)
//!       init acc; acc += off
//!       forchunk col in lane's chunk { acc += input; writes }
//!   ```
//!   The serial depth drops by the lane count; the additions are regrouped, so
//!   the scan declares `Deterministic` instead of `ExactOrder`.
//!
//! A `Read` from a quantized tensor may only appear below `Dequant`. Lowering
//! fuses the pair into a load of an F16 scale and signed byte, with addresses
//! derived from `QuantFormat`. No dequantized tensor is materialized, and
//! emitters do not know the quantized format.
//!
//! V1 limitations are explicit errors: mixed reductions and scans, mixed scan
//! directions, and any pairing of a scan strategy with a reduction strategy
//! that cannot carry it.

pub mod analysis;
pub mod dequant;
pub mod error;
pub mod lowerer;
pub mod strategies;
pub mod vectorize;

#[cfg(test)]
mod tests;

pub(crate) use analysis::{Analysis, analyze};
pub(crate) use dequant::{QuantRead, SegmentHeader, SegmentPlan, plan_segments};
pub use error::{LowerError, can_lower_dequant};
pub(crate) use lowerer::Lowerer;
pub(crate) use strategies::reduction::{Traversal, lower_accum_loop, reduce_levels};
pub(crate) use vectorize::{VecShape, widen_tiled};

use std::collections::HashMap;

use rir_core::{ArgId, Kernel, Op, ReductionSemantics, ValueId};

use crate::loop_ir::*;
use crate::schedule::{ParallelMapping, ReductionStrategy, SUBGROUP_LANES, ScanStrategy, Schedule};

use crate::schedule::check_schedule;
use rir_core::ValidatedKernel;
use strategies::reduction::lower_lanes;
use strategies::scan::{lower_blocked_scan, lower_tiled_scan};
use strategies::serial::lower_serial;
use strategies::tiled::lower_tiled;
use vectorize::lower_vectorized;

/// Applies `schedule` to `kernel`, producing the Loop IR the emitters print.
///
/// Takes a [`ValidatedKernel`] and not a `&Kernel`: the body
/// below indexes `ops`, `axes` and `args` directly and reads operands assuming
/// they are defined, which is legitimate exactly when `validate` has run. It
/// The parameter type is the proof that validation has run, so malformed
/// kernels are rejected before this code indexes their internal tables.
///
/// The coercion goes one way, so the borrow a `ValidatedKernel` hands out is not
/// a way back in:
///
/// ```compile_fail
/// # use rir_core::{Extent, KernelBuilder, TensorType};
/// # let mut b = KernelBuilder::new("copy");
/// # let x = b.input("x", TensorType::f32_2d());
/// # let y = b.output("y", TensorType::f32_2d());
/// # let row = b.axis("row", Extent::Dim { arg: x, dim: 1 });
/// # let col = b.axis("col", Extent::Dim { arg: x, dim: 0 });
/// # let v = b.read(x, &[col, row]);
/// # b.write(y, &[col, row], v);
/// # let k = b.finish().unwrap();
/// // `&Kernel` does not coerce to `&ValidatedKernel`.
/// rir_lower::lower(k.kernel(), rir_lower::Schedule::cpu_serial()).unwrap();
/// ```
///
/// ```
/// # use rir_core::{Extent, KernelBuilder, TensorType};
/// # let mut b = KernelBuilder::new("copy");
/// # let x = b.input("x", TensorType::f32_2d());
/// # let y = b.output("y", TensorType::f32_2d());
/// # let row = b.axis("row", Extent::Dim { arg: x, dim: 1 });
/// # let col = b.axis("col", Extent::Dim { arg: x, dim: 0 });
/// # let v = b.read(x, &[col, row]);
/// # b.write(y, &[col, row], v);
/// let k = b.finish().unwrap();
/// let lk = rir_lower::lower(&k, rir_lower::Schedule::cpu_serial()).unwrap();
/// assert_eq!(lk.name, "copy");
/// ```
///
/// # Errors
///
/// `LowerError::Schedule` if the schedule is malformed or its strategy
/// contradicts a declared reduction semantics, then the first lowering
/// limitation the pair reaches.
pub fn lower(kernel: &ValidatedKernel, schedule: Schedule) -> Result<LoopKernel, LowerError> {
    // The witness has done its work at the signature; everything below reads a
    // plain kernel.
    let kernel: &Kernel = kernel;
    check_schedule(kernel, &schedule)?;
    let a = analyze(kernel)?;

    // Manifest semantics: declared reductions plus exact ordering for each
    // scan, which is sequential in v1.
    let mut semantics: Vec<ReductionSemantics> = kernel
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::Reduce { semantics, .. } => Some(*semantics),
            _ => None,
        })
        .collect();
    // A scan's semantics is a property of the **strategy**, not of the kernel:
    // sequential traversal keeps the exact order, the blocked one regroups the
    // additions in an order the declared lane count fixes.
    let scan_semantics = match schedule.scan {
        ScanStrategy::Serial => ReductionSemantics::ExactOrder,
        // Both lane strategies regroup the additions; what separates them is
        // where the elements are read from, not the order they are combined in.
        ScanStrategy::BlockedLanes | ScanStrategy::TiledLanes { .. } => {
            ReductionSemantics::Deterministic
        }
    };
    semantics.extend(a.scans.iter().map(|_| scan_semantics));

    let mut lo = Lowerer {
        shared: Vec::new(),
        k: kernel,
        var_names: Vec::new(),
        var_kinds: Vec::new(),
        env: HashMap::new(),
        results_env: HashMap::new(),
        axis_vars: HashMap::new(),
        body: Vec::new(),
        allow_results: false,
        linear: None,
    };

    // The linear-addressing claim, decided before a single
    // register is bound because it decides *which* registers exist: under it the
    // axis indices are neither computed by the shader nor bound here, so a body
    // that reads one fails with `AxisOutOfScope` instead of reading a register
    // the flattened nest never assigns.
    let linear_addr = schedule.linear_addr;
    if linear_addr {
        if !schedule.flatten {
            return Err(LowerError::LinearAddrUnsupported {
                why: "flattening required: the linear index is what the address is computed from",
            });
        }
        check_linear_addressable(kernel, &a)?;
    }

    // Parallel-axis variables in declaration order.
    let par_vars: Vec<VarId> = a
        .par_axes
        .iter()
        .map(|&ax| {
            let v = lo.new_var(&kernel.axes()[ax.0 as usize].name, VarKind::Idx);
            if !linear_addr {
                lo.axis_vars.insert(ax, v);
            }
            v
        })
        .collect();

    // The linear index, allocated here only under linear addressing, where it
    // *is* the address and therefore has to exist before the accesses that name
    // it. The decomposing nest allocates its own below, after the body, which is
    // where it has always been - a register order the generated artifacts carry,
    // and one this claim has no reason to renumber.
    let linear = linear_addr.then(|| lo.new_var("flat", VarKind::Idx));
    if let Some(linear) = linear {
        lo.linear = Some((linear, schedule.vector_width.max(1)));
    }

    if schedule.vector_width > 4 {
        return Err(LowerError::VectorWidthUnsupported {
            why: "width exceeds the four native components of GPU emitters",
        });
    }
    let vector = schedule.vector_width;
    // `TiledStage` carries the width itself - there it is register tiling over
    // a contraction, not an elementwise traversal - and states its own
    // conditions in `lower_tiled`.
    if vector > 1 && schedule.reduction != ReductionStrategy::TiledStage {
        // Everything the width needs in order to *mean* something. A width the
        // lowering could not honour must be an error and not a silent 1: the
        // schedule field is the only place the decision is written down, and a
        // shader that ignored it would still be published as `vec4`.
        if schedule.par_map != ParallelMapping::Invocation
            || schedule.reduction != ReductionStrategy::Serial
        {
            return Err(LowerError::VectorWidthUnsupported {
                why: "invocation mapping and Serial strategy required",
            });
        }
        if a.inner_axis.is_some() {
            return Err(LowerError::VectorWidthUnsupported {
                why: "kernel with an inner axis - only the elementwise strip is vectorized",
            });
        }
        if !a.inner_writes.is_empty() {
            return Err(LowerError::VectorWidthUnsupported {
                why: "writes depend on the inner axis",
            });
        }
    }

    // A depth outside the strategy that reads it would be a schedule field
    // nothing honours - the failure mode already closed for `vector_width`.
    if (schedule.tile_depth > 0) != (schedule.reduction == ReductionStrategy::TiledStage) {
        return Err(LowerError::TilingUnsupported {
            why: "tile_depth and strategy must match (either without the other is meaningless)",
        });
    }

    let innermost = match schedule.reduction {
        ReductionStrategy::TiledStage => lower_tiled(&mut lo, &a, &par_vars, &schedule)?,
        // Linear addressing has no tail to branch on: the claim requires the
        // contiguous extent to be a whole number of vectors, so every invocation
        // owns a complete one and `VecTail` would print a branch no invocation
        // can take (`lower_linear`).
        ReductionStrategy::Serial if linear_addr => lower_linear(
            &mut lo,
            &a,
            linear.expect("linear addressing implies a flattened nest"),
            vector,
        )?,
        ReductionStrategy::Serial if vector > 1 => {
            lower_vectorized(&mut lo, &a, par_vars[0], vector)?
        }
        ReductionStrategy::Serial => {
            // `Serial` uses no collective, so reserving a multi-lane workgroup
            // per index would waste every lane but one. A trivial CPU block
            // remains consistent with `Workgroup`.
            if schedule.par_map == ParallelMapping::Workgroup && schedule.block != [1, 1, 1] {
                return Err(LowerError::MappingMismatch {
                    strategy: schedule.reduction,
                    mapping: schedule.par_map,
                });
            }
            // `BlockedLanes` needs a lane collective, which `Serial` does not
            // provide: the pair is a schedule error, not a silent downgrade to
            // the sequential scan.
            if !a.scans.is_empty() && schedule.scan != ScanStrategy::Serial {
                return Err(LowerError::BlockedScanRequiresLanes);
            }
            lower_serial(&mut lo, &a)?
        }
        ReductionStrategy::SubgroupTree
        | ReductionStrategy::SharedTree
        | ReductionStrategy::HierarchicalTree => {
            if schedule.par_map != ParallelMapping::Workgroup {
                return Err(LowerError::MappingMismatch {
                    strategy: schedule.reduction,
                    mapping: schedule.par_map,
                });
            }
            // Lanes are the only way to run a scan here; a scan left on
            // `Serial` while the reduction strategy is collective would silently
            // give every lane the whole row.
            if !a.scans.is_empty() && schedule.scan == ScanStrategy::Serial {
                return Err(LowerError::ScanRequiresSerial);
            }
            if a.inner_axis.is_none() {
                return Err(LowerError::LanesRequireInnerAxis);
            }
            let lanes = schedule.block[0];
            // The tiled scan's prefix over lane totals **is** the shared tree,
            // the same `workgroup_scan_stmts` the blocked scan lowers under
            // `SharedTree`, and it does not consult the reduction strategy.
            // Accepting
            // `SubgroupTree` beside it would publish `subgroup_tree` in the
            // manifest for a kernel that uses none, and would let a lane count
            // that is not a power of two through the check below: the tree would
            // then combine the wrong lanes and the scan would simply be false.
            // The strategy a lowering does not honour is an error, never a
            // relabelling (ADR-2 section 5).
            if matches!(schedule.scan, ScanStrategy::TiledLanes { .. })
                && schedule.reduction != ReductionStrategy::SharedTree
            {
                return Err(LowerError::TiledScanUnsupported {
                    why: "SharedTree reduction required: the prefix of lane totals is the shared \
                          tree, and nothing else implements it",
                });
            }
            if schedule.reduction == ReductionStrategy::SharedTree && !lanes.is_power_of_two() {
                return Err(LowerError::SharedTreeRequiresPowerOfTwo { lanes });
            }
            if schedule.reduction == ReductionStrategy::HierarchicalTree {
                // The two conditions the second stage rests on:
                // a workgroup that is not a whole number
                // of subgroups would leave a partial one whose total nobody
                // stores; more subgroups than one subgroup can reduce would need
                // a third stage, which is a different strategy and not a wider
                // block.
                if !lanes.is_multiple_of(SUBGROUP_LANES) {
                    return Err(LowerError::HierarchicalTreeGeometry {
                        why: "the lane count is not a whole number of subgroups",
                    });
                }
                if lanes / SUBGROUP_LANES > SUBGROUP_LANES {
                    return Err(LowerError::HierarchicalTreeGeometry {
                        why: "more subgroups than one subgroup can reduce: the second stage \
                              would need a third",
                    });
                }
                // A scan recombines through `workgroup_scan_stmts`, which is the
                // shared tree and consults no strategy: accepting it here would
                // publish `hierarchical_tree` for a kernel that uses none - the
                // relabelling ADR-2 forbids.
                if !a.scans.is_empty() {
                    return Err(LowerError::HierarchicalTreeGeometry {
                        why: "kernel with a scan: the prefix of lane totals is the shared tree, \
                              and nothing else implements it",
                    });
                }
            }
            match (a.scans.is_empty(), schedule.scan) {
                // A scan strategy on a kernel with no scan is a field nothing
                // honours - the same refusal `tile_depth` gets above.
                (true, ScanStrategy::TiledLanes { .. }) => {
                    return Err(LowerError::TiledScanUnsupported {
                        why: "kernel without a scan: nothing would honor the strategy",
                    });
                }
                (true, _) => lower_lanes(&mut lo, &a, lanes, schedule.reduction)?,
                (false, ScanStrategy::TiledLanes { items }) => {
                    lower_tiled_scan(&mut lo, &a, lanes, items)?
                }
                (false, _) => lower_blocked_scan(&mut lo, &a, lanes, schedule.reduction)?,
            }
        }
    };

    // The flattened dispatch: one linear index over the
    // whole parallel space instead of the three-dimensional nest below. It is a
    // *replacement* for that nest and not a layer on it, so it is built here and
    // the nest is skipped entirely.
    // `linear` is a register of its own - not any axis's index - because the
    // first decomposition step would otherwise read what it is about to write,
    // and because under linear addressing it is the only index there is.
    let mut body = if schedule.flatten {
        let linear = linear.unwrap_or_else(|| lo.new_var("flat", VarKind::Idx));
        flatten_nest(&schedule, &a, linear, &par_vars, innermost, vector)?
    } else {
        parallel_nest(&schedule, &a, &par_vars, innermost, vector)
    };

    // Strength reduction on the finished nest (ADR-2 section 7). It runs
    // last, on the loop nest every emitter and the interpreter will consume, so
    // the oracle judges the addresses the device actually computes.
    crate::hoist::hoist_invariants(&mut body, &mut lo.var_names, &mut lo.var_kinds);
    // Then the canonicalisation of what hoisting produced.
    // It runs *after* it and not instead of it: hoisting is what brings two
    // identical expressions into the same block - one per sibling loop, one per
    // half of a vectorized traversal - and this is what then computes them once.
    crate::cse::canonicalize(&mut body);

    Ok(LoopKernel {
        name: kernel.name().to_string(),
        args: kernel.args().to_vec(),
        params: kernel.params().to_vec(),
        axes: kernel.axes().to_vec(),
        arg_axes: rir_core::arg_axes(kernel),
        folds: kernel
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::RepeatIndex { index, over } => match kernel.ops()[index.0 as usize] {
                    Op::Index(a) => Some((a, *over)),
                    _ => None,
                },
                _ => None,
            })
            .collect(),
        constraints: kernel.constraints().to_vec(),
        reduction_semantics: semantics,
        var_names: lo.var_names,
        var_kinds: lo.var_kinds,
        shared: lo.shared,
        body,
        schedule,
    })
}

/// The whole parallel space as **one** grid dimension.
///
/// Every condition it needs is checked here rather than assumed, and each of
/// them is a schedule field that would otherwise be silently ignored - the
/// failure mode once associated with `vector_width`. A collective maps its axis to a
/// *workgroup*, so folding that axis into a linear invocation index would put
/// the lanes of one row in different blocks; a staged contraction is the same
/// objection with barriers on top.
fn flatten_nest(
    schedule: &Schedule,
    a: &Analysis,
    linear: VarId,
    par_vars: &[VarId],
    innermost: Vec<Stmt>,
    vector: u32,
) -> Result<Vec<Stmt>, LowerError> {
    // Two shapes of flattening, and the condition is the same fact read twice:
    // the linear index has to name whatever a parallel axis named. Without a
    // collective that is an invocation; with one it is a
    // **workgroup**, whose lanes then cooperate on the point. What is refused
    // is the crossing - a linear *invocation* index under a
    // collective would split one row across blocks, which is the error this
    // guard has always been about.
    let collective = matches!(
        schedule.reduction,
        ReductionStrategy::SubgroupTree
            | ReductionStrategy::SharedTree
            | ReductionStrategy::HierarchicalTree
    );
    let coherent = match schedule.par_map {
        ParallelMapping::Invocation => {
            schedule.reduction == ReductionStrategy::Serial && schedule.scan == ScanStrategy::Serial
        }
        ParallelMapping::Workgroup => collective,
    };
    if !coherent {
        return Err(LowerError::FlattenUnsupported {
            why: "the mapping and the strategy disagree: a collective flattens over workgroups \
                  and everything else over invocations, and a linear invocation index under a \
                  collective would split one row across blocks",
        });
    }
    if a.par_axes.is_empty() {
        return Err(LowerError::FlattenUnsupported {
            why: "no parallel axis to flatten",
        });
    }
    Ok(vec![Stmt::ParallelFlat {
        linear,
        // One invocation per point without a collective, one workgroup per point
        // with one: the same choice `parallel_nest` makes per grid dimension,
        // made once for the single dimension this nest has.
        level: match schedule.par_map {
            ParallelMapping::Workgroup => HwLevel::Grid(0),
            ParallelMapping::Invocation => HwLevel::Global(0),
        },
        axes: a
            .par_axes
            .iter()
            .zip(par_vars)
            .map(|(ax, v)| (*v, *ax))
            .collect(),
        // The host decomposes in either case - the divisors are what the total
        // is a product of - so `axes` above is complete. What linear addressing
        // removes is the *shader*'s copy of that decomposition.
        decompose: !schedule.linear_addr,
        vector,
        body: innermost,
    }])
}

/// The elementwise body under linear addressing: the
/// writes lowered once, then widened around the linear register itself.
///
/// There is no `VecTail` here and that is not an omission. The tail exists
/// because a row that is not a whole number of vectors has a last, partial one;
/// linear addressing claims the contiguous extent *is* a whole number of them,
/// otherwise the flattened index and the element index differ by more than the
/// width and the whole identity collapses - so the branch would print a path no
/// invocation can take. The claim is published, and dispatch refuses the node
/// that would need the tail (`rir_variant_desc.linear_addr`).
fn lower_linear(
    lo: &mut Lowerer,
    a: &Analysis,
    linear: VarId,
    width: u32,
) -> Result<Vec<Stmt>, LowerError> {
    let mut body = lower_writes(lo, &a.row_writes)?;
    if width > 1 {
        vectorize::widen(lo, &mut body, linear, width)?;
    }
    Ok(body)
}

/// The *same shape* half of the linear-addressing claim, proved on the semantic
/// graph.
///
/// `addr = linear · width · elem_bytes` holds exactly when the flattened point
/// and the element the access reads are the same point of the same space. That
/// is four conditions, and each one is a way the identity breaks rather than a
/// convenience:
///
/// - **no inner axis.** The linear index covers the parallel space; a reduction
///   or scan axis is walked *inside* it and is not in the product.
/// - **no axis outside it.** An axis declared for its extent alone - the
///   divisor of a fold - is a dimension the linear index does not count.
/// - **no fold.** A repeated operand is, by definition, a second shape: the
///   claim's other half.
/// - **every access is the identity on that space.** `idx[d]` must be the index
///   of the `d`-th parallel axis, so the stride sum collapses; a permutation, a
///   constant, or a computed position does not.
///
/// A quantized argument is refused with them, and separately: its `nb[0]` is a
/// block and not an element, so `elem_bytes` is not the scale of its address.
fn check_linear_addressable(k: &Kernel, a: &Analysis) -> Result<(), LowerError> {
    if a.inner_axis.is_some() {
        return Err(LowerError::LinearAddrUnsupported {
            why: "kernel with an inner axis: the linear index counts the parallel space alone",
        });
    }
    if a.par_axes.len() != k.axes().len() {
        return Err(LowerError::LinearAddrUnsupported {
            why: "an axis outside the flattened space: its extent is a dimension the linear                   index does not count",
        });
    }
    for op in k.ops() {
        let (tensor, idx) = match op {
            Op::RepeatIndex { .. } => {
                return Err(LowerError::LinearAddrUnsupported {
                    why: "a repeated operand is a second shape, which is what the claim excludes",
                });
            }
            Op::Read { tensor, idx } => (*tensor, idx),
            Op::Write { tensor, idx, .. } => (*tensor, idx),
            _ => continue,
        };
        if matches!(
            k.args()[tensor.0 as usize].ty.dtype,
            rir_core::DType::Quant(_)
        ) {
            return Err(LowerError::LinearAddrUnsupported {
                why: "quantized argument: its contiguous stride is a block, not an element",
            });
        }
        if idx.len() != a.par_axes.len() {
            return Err(LowerError::LinearAddrUnsupported {
                why: "an access that does not index every parallel axis",
            });
        }
        for (d, &iv) in idx.iter().enumerate() {
            match k.ops()[iv.0 as usize] {
                Op::Index(ax) if ax == a.par_axes[d] => {}
                _ => {
                    return Err(LowerError::LinearAddrUnsupported {
                        why: "an access whose index is not the parallel axis of that dimension",
                    });
                }
            }
        }
    }
    Ok(())
}

/// The grid nest: the first `grid_dims` parallel axes to grid dimensions at the
/// level the schedule selects, the rest sequential because a GPU grid has only
/// three dimensions.
fn parallel_nest(
    schedule: &Schedule,
    a: &Analysis,
    par_vars: &[VarId],
    innermost: Vec<Stmt>,
    vector: u32,
) -> Vec<Stmt> {
    let n_grid = a.par_axes.len().min(schedule.grid_dims);
    let mut body = innermost;
    for i in (n_grid..a.par_axes.len()).rev() {
        body = vec![Stmt::For {
            var: par_vars[i],
            axis: a.par_axes[i],
            reverse: false,
            body,
        }];
    }
    for i in (0..n_grid).rev() {
        let level = match schedule.par_map {
            ParallelMapping::Workgroup => HwLevel::Grid(i as u8),
            ParallelMapping::Invocation => HwLevel::Global(i as u8),
        };
        body = vec![Stmt::Parallel {
            var: par_vars[i],
            axis: a.par_axes[i],
            level,
            // Only the first grid dimension is vectorized: it walks the
            // contiguous axis, which is the one on which `w` consecutive
            // elements are `w` consecutive addresses.
            vector: if i == 0 { vector } else { 1 },
            // A staged body holds barriers, so an invocation past the extent
            // may only leave early when *its whole workgroup* is past it,
            // which is exactly the dimensions a workgroup is one index wide on.
            // The rest keep their invocations alive and guard the write
            // instead (`Stmt::InBounds`, built by `lower_tiled`).
            bounded: schedule.reduction != ReductionStrategy::TiledStage || schedule.block[i] <= 1,
            body,
        }];
    }
    body
}

/// Emits a list of writes in the current scope with a fresh memo.
pub(crate) fn lower_writes(
    lo: &mut Lowerer,
    writes: &[(ArgId, Vec<ValueId>, ValueId)],
) -> Result<Vec<Stmt>, LowerError> {
    lo.begin_scope(true);
    for (tensor, idx, value) in writes {
        lo.lower_store(*tensor, idx, *value)?;
    }
    Ok(lo.take_body())
}
