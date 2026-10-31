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

/// A lane tile: the second parallel axis, of which one workgroup owns `count`
/// consecutive indices from `first` (`Schedule::lane_tile`).
#[derive(Clone, Copy)]
pub(crate) struct LaneTile {
    pub(crate) axis: AxisId,
    pub(crate) first: VarId,
    pub(crate) count: u32,
}

/// One level's accumulators under a lane tile: one set per tiled index, the
/// index registers they are lowered at, and the earlier levels' results each
/// index sees.
pub(crate) struct TiledAccums<'a> {
    pub(crate) axis: AxisId,
    /// `first`, `first + 1`, …: one register per tiled index.
    pub(crate) idx: &'a [VarId],
    /// `accs[r]` accumulates at `idx[r]`.
    pub(crate) accs: &'a [Vec<Accum>],
    /// `results[r]` is the results environment index `r` reads.
    pub(crate) results: &'a [HashMap<ValueId, VarId>],
}

/// One accumulator: its register, its operator, the value it accumulates and
/// the reduction it closes.
pub(crate) type Accum = (VarId, rir_core::ReduceOp, ValueId, ValueId);

/// Whether `v` depends on the index of `axis`.
pub(crate) fn depends_on_axis(k: &Kernel, v: ValueId, axis: AxisId) -> bool {
    let direct = k.ops().iter().enumerate().any(|(i, op)| {
        matches!(op, Op::Index(a) if *a == axis) && reaches(k, v, ValueId(i as u32))
    });
    if direct {
        return true;
    }
    // `reaches` deliberately stops at earlier reduction results. Those are
    // still column-dependent values: sharing their registers (or expressions
    // using them) would give every column the first column's result.
    let mut results = Vec::new();
    collect_result_deps(k, v, &mut results);
    results
        .into_iter()
        .any(|result| match &k.ops()[result.0 as usize] {
            Op::Reduce { value, .. } | Op::Scan { value, .. } => depends_on_axis(k, *value, axis),
            _ => unreachable!("collect_result_deps returns only result nodes"),
        })
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
    tile: Option<&TiledAccums>,
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
    let run = lo.reduce_run;
    let runs = if run > 1 {
        if plan.span <= 1 || reads.is_empty() {
            return Err(LowerError::ReduceRunUnsupported {
                why: "no segmented quantized read for the run to divide",
            });
        }
        if !plan.span.is_multiple_of(run) {
            return Err(LowerError::ReduceRunUnsupported {
                why: "the run does not divide the segment",
            });
        }
        let runs = run_reads(lo.k, inner_axis, accs);
        if runs.is_empty() {
            return Err(LowerError::ReduceRunUnsupported {
                why: "no dense operand is read along the reduced axis",
            });
        }
        runs
    } else {
        Vec::new()
    };
    if plan.span <= 1 || reads.is_empty() {
        let col = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col);
        lo.begin_scope(true);
        emit_level(lo, accs, tile)?;
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
    if runs.is_empty() {
        return lower_segments(lo, span, &reads, inner_axis, accs, traversal, tile, &[]);
    }
    // The run half and the element half, into the same accumulators, behind
    // one uniform test on the operands' contiguous stride.
    let unit = lower_segments(lo, span, &reads, inner_axis, accs, traversal, tile, &runs)?;
    let strided = lower_segments(lo, span, &reads, inner_axis, accs, traversal, tile, &[])?;
    let mut args = Vec::new();
    for rv in &runs {
        let Op::Read { tensor, .. } = &lo.k.ops()[rv.0 as usize] else {
            unreachable!("run_reads returns reads only");
        };
        let dense = (*tensor, Lowerer::elem_mem_type(lo.k, *tensor)?);
        if !args.contains(&dense) {
            args.push(dense);
        }
    }
    Ok(vec![Stmt::UnitStride {
        args,
        unit,
        strided,
    }])
}

/// The segment loop of one level: the block headers once per segment, then the
/// element loop - or, given `runs`, the run groups.
#[expect(clippy::too_many_arguments)]
fn lower_segments(
    lo: &mut Lowerer,
    span: u32,
    reads: &[QuantRead],
    inner_axis: AxisId,
    accs: &[Accum],
    traversal: Traversal,
    tile: Option<&TiledAccums>,
    runs: &[ValueId],
) -> Result<Vec<Stmt>, LowerError> {
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
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
    for read in reads {
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

    if !runs.is_empty() {
        body.push(lower_run_groups(
            lo, &heads, runs, seg, span, inner_axis, accs, tile,
        )?);
        out.push(segment_loop(traversal, start, seg, inner_axis, span, body));
        return Ok(out);
    }

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
    emit_level(lo, accs, tile)?;
    let inner_body = lo.take_body();
    body.push(Stmt::ForConst {
        var: elem,
        count: span,
        body: inner_body,
    });

    out.push(segment_loop(traversal, start, seg, inner_axis, span, body));
    Ok(out)
}

/// The loop over a lane's (or the invocation's) segments.
fn segment_loop(
    traversal: Traversal,
    start: Option<VarId>,
    seg: VarId,
    inner_axis: AxisId,
    span: u32,
    body: Vec<Stmt>,
) -> Stmt {
    match (traversal, start) {
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
    }
}

/// The dense reads of a level that a run can serve: an F32 or F16 operand
/// whose first index *is* the reduced axis and whose other indices do not
/// move with it, so element `c` of the run is `c` strides along dimension 0.
///
/// Anything else the level reads along the reduced axis is still read element
/// by element, which is correct and only slower.
fn run_reads(k: &Kernel, inner_axis: AxisId, accs: &[Accum]) -> Vec<ValueId> {
    k.ops()
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            let v = ValueId(i as u32);
            let Op::Read { tensor, idx } = op else {
                return None;
            };
            let dense = matches!(
                k.args()[tensor.0 as usize].ty.dtype,
                rir_core::DType::F32 | rir_core::DType::F16
            );
            let first_is_axis = idx
                .first()
                .is_some_and(|i| matches!(k.ops()[i.0 as usize], Op::Index(a) if a == inner_axis));
            let rest_fixed = idx
                .iter()
                .skip(1)
                .all(|i| !depends_on_axis(k, *i, inner_axis));
            let read_here = accs.iter().any(|(_, _, rin, _)| reaches(k, *rin, v));
            (dense && first_is_axis && rest_fixed && read_here).then_some(v)
        })
        .collect()
}

/// The element loop of a segment when the dense operands are read in runs.
///
/// ```text
/// for g in 0..span/run:                       // constant trip count
///   col_c = seg + g·run + c; w_c = dequant(col_c)   for c in 0..run
///   per tiled index:
///     x_0..x_{run-1} = run of x at col_0       // one access
///     acc += f(w_0, x_0, …); …; acc += f(w_{run-1}, x_{run-1}, …)
/// ```
///
/// Each accumulator still receives its terms in column order, one per element,
/// so the sum is the one the element loop computes; what changes is that a
/// dense operand is read once per run instead of once per element. The decoded
/// weights do not depend on the tiled axis and are computed once for every
/// index, as in the element loop.
#[expect(clippy::too_many_arguments)]
fn lower_run_groups(
    lo: &mut Lowerer,
    heads: &[SegmentHeader],
    runs: &[ValueId],
    seg: VarId,
    span: u32,
    inner_axis: AxisId,
    accs: &[Accum],
    tile: Option<&TiledAccums>,
) -> Result<Stmt, LowerError> {
    let run = lo.reduce_run;
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let group = lo.new_var(&format!("{inner_name}_g"), VarKind::Idx);
    lo.begin_scope(true);
    let e0 = lo.emit("e0", VarKind::Idx, LExpr::IMulC(group, run));
    let mut cols = Vec::new();
    let mut seeds: Vec<(ValueId, Vec<VarId>)> =
        heads.iter().map(|h| (h.read.value, Vec::new())).collect();
    for c in 0..run {
        let elem = if c == 0 {
            e0
        } else {
            lo.emit("e", VarKind::Idx, LExpr::IAddC(e0, c))
        };
        cols.push(lo.emit(&inner_name, VarKind::Idx, LExpr::IAdd(seg, elem)));
        for (head, (_, values)) in heads.iter().zip(seeds.iter_mut()) {
            let inb = lo.emit("inb", VarKind::Idx, LExpr::IAdd(head.inb0, elem));
            values.push(lo.lower_dequant_element(
                head.read.arg,
                head.read.format,
                &head.base,
                &head.header,
                inb,
            )?);
        }
    }
    let ctx = RunCtx {
        inner_axis,
        cols,
        seeds,
        runs,
    };
    match tile {
        None => emit_run_accums(lo, accs, &ctx)?,
        Some(tile) => emit_tiled_with(lo, tile, &mut |lo, accs| emit_run_accums(lo, accs, &ctx))?,
    }
    let body = lo.take_body();
    Ok(Stmt::ForConst {
        var: group,
        count: span / run,
        body,
    })
}

/// What one run group resolved before its accumulations: the column of every
/// element, the decoded weight of every element, and the dense reads to serve
/// in runs.
struct RunCtx<'a> {
    inner_axis: AxisId,
    cols: Vec<VarId>,
    seeds: Vec<(ValueId, Vec<VarId>)>,
    runs: &'a [ValueId],
}

/// One index's accumulations over a run group: the runs read at the group's
/// first column, then the level once per element, each seeing its own column,
/// weight and run element.
///
/// Every value that depends on the reduced axis is dropped from the memo
/// between elements, or element `c` would reuse a register computed at `c - 1`.
fn emit_run_accums(lo: &mut Lowerer, accs: &[Accum], ctx: &RunCtx) -> Result<(), LowerError> {
    let k = lo.k;
    let inner = ctx.inner_axis;
    let keep = |env: &HashMap<ValueId, VarId>| -> HashMap<ValueId, VarId> {
        env.iter()
            .filter(|(v, _)| !depends_on_axis(k, **v, inner))
            .map(|(v, r)| (*v, *r))
            .collect()
    };
    lo.env = keep(&lo.env);
    lo.axis_vars.insert(inner, ctx.cols[0]);
    let mut run_regs = Vec::with_capacity(ctx.runs.len());
    for &rv in ctx.runs {
        let Op::Read { tensor, idx } = k.ops()[rv.0 as usize].clone() else {
            unreachable!("run_reads returns reads only");
        };
        let idx_vars = match lo.linear {
            Some(_) => Vec::new(),
            None => lo.lower_idx(&idx)?,
        };
        let dsts: Vec<VarId> = (0..ctx.cols.len())
            .map(|_| lo.new_var("t", VarKind::F32))
            .collect();
        lo.body.push(Stmt::LoadRun {
            dsts: dsts.clone(),
            arg: tensor,
            ty: Lowerer::elem_mem_type(k, tensor)?,
            addr: lo.dense_addr(tensor, &idx_vars),
        });
        run_regs.push(dsts);
    }
    let base = keep(&lo.env);
    for (c, &col) in ctx.cols.iter().enumerate() {
        lo.env = base.clone();
        lo.axis_vars.insert(inner, col);
        for (v, regs) in &ctx.seeds {
            lo.env.insert(*v, regs[c]);
        }
        for (v, regs) in ctx.runs.iter().zip(&run_regs) {
            lo.env.insert(*v, regs[c]);
        }
        emit_accums(lo, accs)?;
    }
    lo.env = keep(&lo.env);
    Ok(())
}

/// The accumulations of one level, tiled or not.
fn emit_level(
    lo: &mut Lowerer,
    accs: &[(VarId, rir_core::ReduceOp, ValueId, ValueId)],
    tile: Option<&TiledAccums>,
) -> Result<(), LowerError> {
    match tile {
        None => emit_accums(lo, accs),
        Some(tile) => emit_tiled_accums(lo, tile),
    }
}

/// The accumulations of one level once per tiled index.
///
/// The first index is lowered as the untiled level is, and everything it
/// computed that does not depend on the tiled axis - the decoded weight, its
/// address, the scale of its block - stays in the memo for the others, which
/// is the whole point: one read and one decode for `count` products. Each
/// further index is guarded (`Stmt::InBounds`), since the last tile of an axis
/// may be partial and its reads must not leave the tensor; the first needs no
/// guard because the grid already returned past the axis.
fn emit_tiled_accums(lo: &mut Lowerer, tile: &TiledAccums) -> Result<(), LowerError> {
    emit_tiled_with(lo, tile, &mut emit_accums)
}

/// The accumulations of one index, as `emit_tiled_with` asks for them.
type EmitIndex<'f> = dyn FnMut(&mut Lowerer, &[Accum]) -> Result<(), LowerError> + 'f;

/// `emit_tiled_accums` with the accumulations of one index supplied by the
/// caller: the plain level, or a run group's.
fn emit_tiled_with(
    lo: &mut Lowerer,
    tile: &TiledAccums,
    emit: &mut EmitIndex<'_>,
) -> Result<(), LowerError> {
    let k = lo.k;
    let mut shared: HashMap<ValueId, VarId> = HashMap::new();
    for (r, (&idx, accs)) in tile.idx.iter().zip(tile.accs).enumerate() {
        lo.axis_vars.insert(tile.axis, idx);
        lo.results_env = tile.results[r].clone();
        if r == 0 {
            emit(lo, accs)?;
            shared = lo
                .env
                .iter()
                .filter(|(v, _)| !depends_on_axis(k, **v, tile.axis))
                .map(|(v, r)| (*v, *r))
                .collect();
            continue;
        }
        let outer = lo.take_body();
        lo.env = shared.clone();
        emit(lo, accs)?;
        let guarded = lo.take_body();
        lo.body = outer;
        lo.body.push(Stmt::InBounds {
            bounds: vec![(idx, tile.axis)],
            body: guarded,
        });
    }
    lo.axis_vars.insert(tile.axis, tile.idx[0]);
    lo.env = shared;
    lo.results_env = tile.results[0].clone();
    Ok(())
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
    tile: Option<LaneTile>,
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

    // Under a lane tile, one index register per tiled index - the first is the
    // grid's own - and one results environment per index, which the levels
    // below fill and the writes read.
    let tile_idx: Vec<VarId> = match tile {
        None => Vec::new(),
        Some(t) => {
            let mut idx = vec![t.first];
            for r in 1..t.count {
                let v = lo.new_var(
                    &format!("{}{r}", lo.k.axes()[t.axis.0 as usize].name),
                    VarKind::Idx,
                );
                lane_body.push(Stmt::Compute(Inst {
                    dst: v,
                    expr: LExpr::IAddC(t.first, r),
                }));
                idx.push(v);
            }
            idx
        }
    };
    let mut tile_results: Vec<HashMap<ValueId, VarId>> =
        vec![lo.results_env.clone(); tile_idx.len()];

    if !a.reduces.is_empty() {
        let plan = plan_segments(lo.k, inner_axis);
        // A quantized read whose position moves with the tiled axis would have
        // its block header decoded once, at the first index, and reused at the
        // others. Nothing tiled reads one; refusing it is what keeps that true.
        if let Some(t) = tile
            && plan
                .reads
                .iter()
                .any(|r| depends_on_axis(lo.k, r.value, t.axis))
        {
            return Err(LowerError::LaneTileUnsupported {
                why: "a quantized read depends on the tiled axis",
            });
        }
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
            // The tiled indices past the first get accumulators of their own.
            let mut tiled_accs: Vec<Vec<Accum>> = Vec::new();
            if !tile_idx.is_empty() {
                tiled_accs.push(accs.clone());
                for r in 1..tile_idx.len() {
                    let mut row = Vec::new();
                    for (rv, rop, rin) in group {
                        let acc = lo.new_var(&format!("acc{}_{r}", rv.0), VarKind::F32);
                        lane_body.push(Stmt::InitAcc { acc, op: *rop });
                        row.push((acc, *rop, *rin, *rv));
                    }
                    tiled_accs.push(row);
                }
            }
            let tiled = tile.map(|t| TiledAccums {
                axis: t.axis,
                idx: &tile_idx,
                accs: &tiled_accs,
                results: &tile_results,
            });

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
                tiled.as_ref(),
            )?);

            // The collectives of one level, together. `SubgroupTree` reduces
            // through a subgroup instruction and has no barrier to share, so it
            // keeps one statement per accumulator; `SharedTree` expands a tree
            // in shared memory, and grouping is what makes its levels - and its
            // barriers - serve every accumulator at once.
            let mut reds = Vec::new();
            let per_index: Vec<(usize, Accum)> = if tiled_accs.is_empty() {
                accs.iter().map(|acc| (0, *acc)).collect()
            } else {
                tiled_accs
                    .iter()
                    .enumerate()
                    .flat_map(|(r, row)| row.iter().map(move |acc| (r, *acc)))
                    .collect()
            };
            for (r, (acc, rop, _, rv)) in &per_index {
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
                if tile_idx.is_empty() {
                    lo.results_env.insert(*rv, red);
                } else {
                    tile_results[*r].insert(*rv, red);
                }
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
        let stmts = match tile {
            None => lower_writes(lo, &a.row_writes)?,
            // One write per tiled index, each reading its own results, and
            // every index past the first guarded like its accumulation.
            Some(t) => {
                let mut stmts = Vec::new();
                for (r, &idx) in tile_idx.iter().enumerate() {
                    lo.axis_vars.insert(t.axis, idx);
                    lo.results_env = tile_results[r].clone();
                    let writes = lower_writes(lo, &a.row_writes)?;
                    if r == 0 {
                        stmts.extend(writes);
                    } else {
                        stmts.push(Stmt::InBounds {
                            bounds: vec![(idx, t.axis)],
                            body: writes,
                        });
                    }
                }
                stmts
            }
        };
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
