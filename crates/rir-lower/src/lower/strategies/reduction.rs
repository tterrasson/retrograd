//! Reductions: the accumulator loop shared by every strategy, the
//! dependency levels it is replayed for, and the lane strategy.

use std::collections::HashMap;

use rir_core::{AxisId, Kernel, Op, ValueId};

use crate::loop_ir::*;
use crate::schedule::{ReductionStrategy, SUBGROUP_LANES};

use crate::lower::{
    Analysis, LowerError, Lowerer, QuantRead, SegmentHeader, SegmentPlan, lower_writes,
    plan_segments,
};

/// Whether `target` is reachable from `from` in the value graph - "does this
/// expression read that node".
///
/// It stops at `Reduce`/`Scan` for the same reason `collect_result_deps` does:
/// a result node is a value of an *earlier* pass, so what it reads is not read
/// by the expression being lowered here.
pub(crate) fn reaches(k: &Kernel, from: ValueId, target: ValueId) -> bool {
    if from == target {
        return true;
    }
    let any = |vs: &[ValueId]| vs.iter().any(|&v| reaches(k, v, target));
    match &k.ops()[from.0 as usize] {
        Op::ConstF32(_) | Op::Param(_) | Op::Index(_) | Op::AxisExtent(_) => false,
        Op::Reduce { .. } | Op::Scan { .. } => false,
        Op::Read { idx, .. } => any(idx),
        Op::Dequant { value, .. } => reaches(k, *value, target),
        Op::RepeatIndex { index, .. } => reaches(k, *index, target),
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => any(&[*a, *b]),
        Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) => reaches(k, *a, target),
        Op::Cmp { lhs, rhs, .. } => any(&[*lhs, *rhs]),
        Op::Select { cond, t, f } => any(&[*cond, *t, *f]),
        Op::Write { .. } => unreachable!("Write is not a value"),
    }
}

/// How the reduced axis is walked by one invocation or one lane.
#[derive(Clone, Copy)]
pub(crate) enum Traversal {
    /// The whole axis, in order (`Serial`).
    Whole,
    /// Interleaved across lanes: lane `l` starts at `l` and steps by `lanes`.
    Lanes { lane: VarId, lanes: u32 },
}

/// Builds the accumulation loop of one reduction level - the traversal, and the
/// segment split when the plan gives one.
///
/// Segmented, the nest is
///
/// ```text
/// for seg = lane·span; seg < n; seg += lanes·span:   // one block header
///   blk, inb0, scale[, dmin, sub-scale]              // read once
///   for e in 0..span:                               // constant trip count
///     col = seg + e; acc += f(dequant(col), …)
/// ```
///
/// and unsegmented it is the loop it has always been. The two halves are the
/// **same** accumulation lowered around a different index register, so neither
/// is a transcription of the other - and the decoder is the same
/// `lower_dequant_header`/`lower_dequant_element` pair a direct read and a
/// staged tile go through, so no format formula is written twice.
pub(crate) fn lower_accum_loop(
    lo: &mut Lowerer,
    plan: &SegmentPlan,
    inner_axis: AxisId,
    accs: &[(VarId, rir_core::ReduceOp, ValueId, ValueId)],
    traversal: Traversal,
) -> Result<Vec<Stmt>, LowerError> {
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    // Only the reads *this* level accumulates. A header emitted for a read the
    // level does not use would be one load per segment nobody reads, and the
    // pruning pass of `crate::cse` drops dead computations, not dead accesses.
    let reads: Vec<QuantRead> = plan
        .reads
        .iter()
        .filter(|r| {
            accs.iter()
                .any(|(_, _, rin, _)| reaches(lo.k, *rin, r.value))
        })
        .cloned()
        .collect();
    if plan.span <= 1 || reads.is_empty() {
        let col = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col);
        lo.begin_scope(true);
        emit_accums(lo, accs)?;
        let body = lo.take_body();
        return Ok(vec![match traversal {
            Traversal::Whole => Stmt::For {
                var: col,
                axis: inner_axis,
                reverse: false,
                body,
            },
            Traversal::Lanes { lane, lanes } => Stmt::ForStrided {
                var: col,
                axis: inner_axis,
                start: lane,
                step: lanes,
                body,
            },
        }]);
    }

    let span = plan.span;
    // The loop's starting index, computed before it: `lane · span`, so a lane's
    // segments are aligned on the span and its blocks are its own.
    let mut out = Vec::new();
    let start = match traversal {
        Traversal::Whole => None,
        Traversal::Lanes { lane, .. } => {
            let v = lo.new_var("seg_start", VarKind::Idx);
            out.push(Stmt::Compute(Inst {
                dst: v,
                expr: LExpr::IMulC(lane, span),
            }));
            Some(v)
        }
    };

    let seg = lo.new_var(&format!("{inner_name}_seg"), VarKind::Idx);
    lo.begin_scope(true);
    // The reduced axis is *not* bound in the segment scope: nothing outside the
    // element loop may read it, and leaving it unbound is what makes an index
    // that depends on it fail loudly instead of silently taking the segment's
    // first element.
    lo.axis_vars.remove(&inner_axis);
    let mut heads = Vec::new();
    for read in &reads {
        let be = read.format.desc().block_elements;
        let block = lo.emit("blk", VarKind::Idx, LExpr::IDivC(seg, be));
        let inb0 = lo.emit("inb0", VarKind::Idx, LExpr::IModC(seg, be));
        // Slot 0 of the index list is the block, which the header addresses
        // itself; the outer indices are ordinary registers of this scope.
        let mut idx_vars = vec![block];
        for &iv in read.idx.iter().skip(1) {
            idx_vars.push(lo.lower_value(iv)?);
        }
        let (base, header) =
            lo.lower_dequant_header(read.arg, &idx_vars, block, Some((inb0, span)), read.format)?;
        heads.push(SegmentHeader {
            read: read.clone(),
            base,
            header,
            inb0,
        });
    }
    let mut body = lo.take_body();

    let elem = lo.new_var(&format!("{inner_name}_e"), VarKind::Idx);
    lo.begin_scope(true);
    let col = lo.emit(&inner_name, VarKind::Idx, LExpr::IAdd(seg, elem));
    lo.axis_vars.insert(inner_axis, col);
    for head in &heads {
        let inb = lo.emit("inb", VarKind::Idx, LExpr::IAdd(head.inb0, elem));
        let value = lo.lower_dequant_element(
            head.read.arg,
            head.read.format,
            &head.base,
            &head.header,
            inb,
        )?;
        // Seeding the memo is what makes the rest of the body the *same*
        // expression the unsegmented lowering builds - the same trick the tiled
        // contraction uses for a staged read.
        lo.env.insert(head.read.value, value);
    }
    emit_accums(lo, accs)?;
    let inner_body = lo.take_body();
    body.push(Stmt::ForConst {
        var: elem,
        count: span,
        body: inner_body,
    });

    out.push(match (traversal, start) {
        // `ForTiled` is the strided-from-zero loop the whole-axis traversal
        // needs, and it is already the one the tiled contraction walks its
        // rounds with: `seg = 0; seg < n; seg += span`.
        (Traversal::Whole, _) => Stmt::ForTiled {
            var: seg,
            axis: inner_axis,
            step: span,
            body,
        },
        (Traversal::Lanes { lanes, .. }, Some(start)) => Stmt::ForStrided {
            var: seg,
            axis: inner_axis,
            start,
            step: lanes * span,
            body,
        },
        (Traversal::Lanes { .. }, None) => unreachable!("lane traversal without a start"),
    });
    Ok(out)
}

/// The accumulations of one level, in the current scope.
pub(crate) fn emit_accums(
    lo: &mut Lowerer,
    accs: &[(VarId, rir_core::ReduceOp, ValueId, ValueId)],
) -> Result<(), LowerError> {
    for (acc, rop, rin, _) in accs {
        let v = lo.lower_value(*rin)?;
        lo.body.push(Stmt::Accum {
            acc: *acc,
            op: *rop,
            value: v,
        });
    }
    Ok(())
}

pub(crate) type Reduces = [(ValueId, rir_core::ReduceOp, ValueId)];

/// Groups kernel reductions by dependency **level**, from lowest to highest.
/// A reduction whose input depends on another runs in a later pass. Both
/// strategies share this grouping because levels are a graph property, not a
/// hardware property.
pub(crate) fn reduce_levels(
    k: &Kernel,
    reduces: &Reduces,
) -> std::collections::BTreeMap<u32, Vec<(ValueId, rir_core::ReduceOp, ValueId)>> {
    let mut levels = std::collections::BTreeMap::new();
    let mut memo = HashMap::new();
    for (rv, rop, rin) in reduces {
        let l = reduce_level(k, reduces, &mut memo, *rv);
        levels
            .entry(l)
            .or_insert_with(Vec::new)
            .push((*rv, *rop, *rin));
    }
    levels
}

/// Finds `Reduce`/`Scan` nodes reachable from `v`, stopping at those nodes.
pub(crate) fn collect_result_deps(k: &Kernel, v: ValueId, out: &mut Vec<ValueId>) {
    match &k.ops()[v.0 as usize] {
        Op::Reduce { .. } | Op::Scan { .. } => out.push(v),
        Op::ConstF32(_) | Op::Param(_) | Op::Index(_) | Op::AxisExtent(_) => {}
        Op::Read { idx, .. } => {
            for &i in idx {
                collect_result_deps(k, i, out);
            }
        }
        Op::Dequant { value, .. } => collect_result_deps(k, *value, out),
        Op::RepeatIndex { index, .. } => collect_result_deps(k, *index, out),
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
            collect_result_deps(k, *a, out);
            collect_result_deps(k, *b, out);
        }
        Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) => collect_result_deps(k, *a, out),
        Op::Cmp { lhs, rhs, .. } => {
            collect_result_deps(k, *lhs, out);
            collect_result_deps(k, *rhs, out);
        }
        Op::Select { cond, t, f } => {
            collect_result_deps(k, *cond, out);
            collect_result_deps(k, *t, out);
            collect_result_deps(k, *f, out);
        }
        Op::Write { .. } => unreachable!("Write is not a value"),
    }
}

/// Reduction level: one plus the maximum level of reductions on which its
/// input depends. SSA form prevents cycles.
pub(crate) fn reduce_level(
    k: &Kernel,
    reduces: &[(ValueId, rir_core::ReduceOp, ValueId)],
    memo: &mut HashMap<ValueId, u32>,
    rv: ValueId,
) -> u32 {
    if let Some(&l) = memo.get(&rv) {
        return l;
    }
    let input = reduces
        .iter()
        .find(|(v, _, _)| *v == rv)
        .map(|(_, _, i)| *i)
        .expect("known reduction");
    let mut deps = Vec::new();
    collect_result_deps(k, input, &mut deps);
    let l = 1 + deps
        .iter()
        .map(|&d| reduce_level(k, reduces, memo, d))
        .max()
        .unwrap_or(0);
    memo.insert(rv, l);
    l
}

pub(crate) fn lower_lanes(
    lo: &mut Lowerer,
    a: &Analysis,
    lanes: u32,
    strategy: ReductionStrategy,
) -> Result<Vec<Stmt>, LowerError> {
    // The hierarchical tree's second parameter, decided once here rather than at
    // every use: `SUBGROUP_LANES` is the width the whole crate already assumes
    // for a lane collective, and `lower` has checked the lane count against it
    // before this function is reached.
    let subgroup = (strategy == ReductionStrategy::HierarchicalTree).then_some(SUBGROUP_LANES);
    let mut lane_body = Vec::new();
    let inner_axis = a.inner_axis.expect("lanes: inner axis required");
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let lane_var = lo.new_var("lane", VarKind::Idx);

    if !a.reduces.is_empty() {
        let plan = plan_segments(lo.k, inner_axis);
        // Use the same dependency **levels** as the sequential path. Each
        // level is a strided loop followed by its collectives; because
        // `subgroupAdd` broadcasts, the next level sees each result in every
        // lane.
        for group in reduce_levels(lo.k, &a.reduces).values() {
            let mut accs = Vec::new();
            for (rv, rop, rin) in group {
                let acc = lo.new_var(&format!("acc{}", rv.0), VarKind::F32);
                lane_body.push(Stmt::InitAcc { acc, op: *rop });
                accs.push((acc, *rop, *rin, *rv));
            }

            // Results from earlier levels are visible. Referring to a level
            // not yet computed yields `NestedReduce` through an environment
            // lookup miss.
            lane_body.extend(lower_accum_loop(
                lo,
                &plan,
                inner_axis,
                &accs,
                Traversal::Lanes {
                    lane: lane_var,
                    lanes,
                },
            )?);

            // The collectives of one level, together. `SubgroupTree` reduces
            // through a subgroup instruction and has no barrier to share, so it
            // keeps one statement per accumulator; `SharedTree` expands a tree
            // in shared memory, and grouping is what makes its levels - and its
            // barriers - serve every accumulator at once.
            let mut reds = Vec::new();
            for (acc, rop, _, rv) in &accs {
                let red = lo.new_var(&format!("red{}", rv.0), VarKind::F32);
                match strategy {
                    ReductionStrategy::SubgroupTree => lane_body.push(Stmt::LaneReduce {
                        op: *rop,
                        src: *acc,
                        dst: red,
                    }),
                    ReductionStrategy::SharedTree | ReductionStrategy::HierarchicalTree => {
                        // One array per accumulator, even grouped: what the
                        // group shares is the barriers, not the storage.
                        //
                        // Its length is the lane count for the flat tree, whose
                        // first level reads every lane's slot, and the *subgroup
                        // count* for the hierarchical one, which never stores
                        // more than one total per subgroup.
                        // 256 lanes are 1 KiB of threadgroup storage per
                        // accumulator against 32 bytes.
                        lo.shared.push((red, subgroup.map_or(lanes, |w| lanes / w)));
                        reds.push(GroupRed {
                            op: *rop,
                            src: *acc,
                            dst: red,
                        })
                    }
                    _ => unreachable!("lower_lanes is called only for a lane strategy"),
                }
                lo.results_env.insert(*rv, red);
            }
            if !reds.is_empty() {
                lane_body.push(Stmt::WorkgroupReduce {
                    reds,
                    lanes,
                    subgroup,
                });
            }
        }
    }

    if !a.row_writes.is_empty() {
        lo.axis_vars.remove(&inner_axis);
        let stmts = lower_writes(lo, &a.row_writes)?;
        lane_body.push(Stmt::LaneZero {
            lane: lane_var,
            body: stmts,
        });
    }

    if !a.inner_writes.is_empty() {
        let col_var = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col_var);
        let phase2 = lower_writes(lo, &a.inner_writes)?;
        lane_body.push(Stmt::ForStrided {
            var: col_var,
            axis: inner_axis,
            start: lane_var,
            step: lanes,
            body: phase2,
        });
    }

    Ok(vec![Stmt::ParallelLane {
        var: lane_var,
        lanes,
        body: lane_body,
    }])
}
