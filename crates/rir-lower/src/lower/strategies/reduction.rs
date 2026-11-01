//! Reductions: the accumulator loop shared by every strategy, the
//! dependency levels it is replayed for, and the lane strategy.

use std::collections::HashMap;

use rir_core::{AxisId, Kernel, Op, ValueId};

use super::stage::{StageB, StagedRead, stage_b_tiles};
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
        Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) | Op::Erf(a) => reaches(k, *a, target),
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

/// A row tile: the first parallel axis, of which each subgroup owns
/// `per_subgroup` consecutive indices and the workgroup `subgroups` times as
/// many, from `first` (`Schedule::row_tile`, `Schedule::subgroups`).
#[derive(Clone, Copy)]
pub(crate) struct RowTile {
    pub(crate) axis: AxisId,
    pub(crate) first: VarId,
    pub(crate) per_subgroup: u32,
    pub(crate) subgroups: u32,
}

/// The points of the parallel space one subgroup accumulates: rows of the
/// first axis by columns of the second, one set of accumulators per point.
///
/// The two axes are not guarded alike, and the difference is the operand each
/// one indexes. A column past the extent would read the other operand past its
/// end, so every column but the first is guarded (`Stmt::InBounds`). A row is
/// read at a register **wrapped** into the axis - a row that exists - so its
/// accumulation needs no guard, and only its write has one.
pub(crate) struct PointTile<'a> {
    /// The first axis, and the register each row's reads are lowered at.
    pub(crate) rows: Option<(AxisId, &'a [VarId])>,
    /// The second axis, and one register per column.
    pub(crate) cols: Option<(AxisId, &'a [VarId])>,
    /// `accs[r][c]` accumulates at row `r`, column `c`.
    pub(crate) accs: &'a [Vec<Vec<Accum>>],
    /// `results[r][c]` is the results environment that point reads.
    pub(crate) results: &'a [Vec<HashMap<ValueId, VarId>>],
}

impl PointTile<'_> {
    fn n_rows(&self) -> usize {
        self.accs.len()
    }

    fn n_cols(&self) -> usize {
        self.accs[0].len()
    }

    fn row_axis(&self) -> Option<AxisId> {
        self.rows.map(|(axis, _)| axis)
    }

    fn col_axis(&self) -> Option<AxisId> {
        self.cols.map(|(axis, _)| axis)
    }

    /// Binds the indices of point `(r, c)` and the results it reads.
    fn bind(&self, lo: &mut Lowerer, r: usize, c: usize) {
        if let Some((axis, regs)) = self.rows {
            lo.axis_vars.insert(axis, regs[r]);
        }
        if let Some((axis, regs)) = self.cols {
            lo.axis_vars.insert(axis, regs[c]);
        }
        lo.results_env = self.results[r][c].clone();
    }
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
///
/// `tile` is the points one subgroup accumulates; `None` is the one point the
/// enclosing scope already binds, accumulated into `accs`.
pub(crate) fn lower_accum_loop(
    lo: &mut Lowerer,
    plan: &SegmentPlan,
    inner_axis: AxisId,
    accs: &[Accum],
    traversal: Traversal,
    tile: Option<&PointTile>,
) -> Result<Vec<Stmt>, LowerError> {
    let single_accs = [vec![accs.to_vec()]];
    let single_results = [vec![lo.results_env.clone()]];
    let single = PointTile {
        rows: None,
        cols: None,
        accs: &single_accs,
        results: &single_results,
    };
    let tile = tile.unwrap_or(&single);

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
        if lo.stage.is_some() {
            return Err(LowerError::StageUnsupported {
                why: "a segmented quantized read is required: the staged tile is one round of \
                      the segment loop",
            });
        }
        let col = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col);
        lo.begin_scope(true);
        emit_points(lo, tile, &mut |lo, _, _, accs| emit_accums(lo, accs))?;
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
        return lower_segments(lo, span, &reads, inner_axis, traversal, tile, &[]);
    }
    // The run half and the element half, into the same accumulators, behind
    // one uniform test on the operands' contiguous stride.
    let unit = lower_segments(lo, span, &reads, inner_axis, traversal, tile, &runs)?;
    let strided = lower_segments(lo, span, &reads, inner_axis, traversal, tile, &[])?;
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

/// The segment loop of one level: the block headers once per segment and per
/// row of the tile, then the element loop - or, given `runs`, a row tile or a
/// factored scale, the group loop.
fn lower_segments(
    lo: &mut Lowerer,
    span: u32,
    reads: &[QuantRead],
    inner_axis: AxisId,
    traversal: Traversal,
    tile: &PointTile,
    runs: &[ValueId],
) -> Result<Vec<Stmt>, LowerError> {
    let k = lo.k;
    let inner_name = k.axes()[inner_axis.0 as usize].name.clone();
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
    // Under `stage_b` the loop variable is the **round's** base on the reduced
    // axis and the segment loop becomes a round loop around a staged tile: each
    // lane accumulates at `round + lane·span + e`, so the body is built at that
    // register rather than at the loop variable's.
    let stage = lo.stage.clone();
    let seg_body = match (&stage, traversal, start) {
        (Some(_), Traversal::Lanes { .. }, Some(_)) => lo.new_var("seg_lane", VarKind::Idx),
        (Some(_), _, _) => {
            return Err(LowerError::StageUnsupported {
                why: "lane traversal required: one round is split across the workgroup's lanes",
            });
        }
        (None, _, _) => seg,
    };
    // One header per read and per row of the tile - the block of a weight row
    // is its own - except for a read the row does not move, whose header the
    // first row already holds.
    let row_axis = tile.row_axis();
    let moves = |read: &QuantRead| row_axis.is_some_and(|ax| depends_on_axis(k, read.value, ax));
    let mut heads: Vec<Vec<SegmentHeader>> = Vec::with_capacity(tile.n_rows());
    for r in 0..tile.n_rows() {
        if let Some((axis, regs)) = tile.rows {
            lo.env.retain(|v, _| !depends_on_axis(k, *v, axis));
            lo.axis_vars.insert(axis, regs[r]);
        }
        let mut row = Vec::with_capacity(reads.len());
        for (i, read) in reads.iter().enumerate() {
            if r > 0 && !moves(read) {
                row.push(heads[0][i].clone());
                continue;
            }
            let be = read.format.desc().block_elements;
            let block = lo.emit("blk", VarKind::Idx, LExpr::IDivC(seg_body, be));
            let inb0 = lo.emit("inb0", VarKind::Idx, LExpr::IModC(seg_body, be));
            // Slot 0 of the index list is the block, which the header addresses
            // itself; the outer indices are ordinary registers of this scope.
            let mut idx_vars = vec![block];
            for &iv in read.idx.iter().skip(1) {
                idx_vars.push(lo.lower_value(iv)?);
            }
            let (base, header) = lo.lower_dequant_header(
                read.arg,
                &idx_vars,
                block,
                Some((inb0, span)),
                read.format,
            )?;
            row.push(SegmentHeader {
                read: read.clone(),
                base,
                header,
                inb0,
            });
        }
        heads.push(row);
    }
    let mut body = lo.take_body();

    if !runs.is_empty() || tile.rows.is_some() || lo.factor_scales {
        let run = if runs.is_empty() { 1 } else { lo.reduce_run };
        body.extend(lower_groups(
            lo,
            &heads,
            runs,
            seg_body,
            span,
            run,
            inner_axis,
            tile,
            stage.as_ref().map(|st| (st, seg)),
        )?);
        out.push(round_loop(
            lo,
            stage.as_ref(),
            traversal,
            start,
            seg,
            seg_body,
            inner_axis,
            span,
            body,
        )?);
        return Ok(out);
    }

    let elem = lo.new_var(&format!("{inner_name}_e"), VarKind::Idx);
    lo.begin_scope(true);
    let col = lo.emit(&inner_name, VarKind::Idx, LExpr::IAdd(seg_body, elem));
    lo.axis_vars.insert(inner_axis, col);
    for head in &heads[0] {
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
    emit_points(lo, tile, &mut |lo, _, _, accs| emit_accums(lo, accs))?;
    let inner_body = lo.take_body();
    body.push(Stmt::ForConst {
        var: elem,
        count: span,
        body: inner_body,
    });

    out.push(round_loop(
        lo,
        stage.as_ref(),
        traversal,
        start,
        seg,
        seg_body,
        inner_axis,
        span,
        body,
    )?);
    Ok(out)
}

/// The loop over a lane's (or the invocation's) segments, or - under
/// `stage_b` - the loop over the workgroup's **rounds**: the loop variable
/// becomes the round's base on the reduced axis, the shared operand's tile is
/// loaded by the whole workgroup between two barriers, and the body runs per
/// lane at `round + lane·span`, held back by the round's own bound.
#[expect(clippy::too_many_arguments)]
fn round_loop(
    lo: &mut Lowerer,
    stage: Option<&StageB>,
    traversal: Traversal,
    start: Option<VarId>,
    seg: VarId,
    seg_body: VarId,
    inner_axis: AxisId,
    span: u32,
    body: Vec<Stmt>,
) -> Result<Stmt, LowerError> {
    let Some(st) = stage else {
        return Ok(segment_loop(traversal, start, seg, inner_axis, span, body));
    };
    let step = match traversal {
        Traversal::Lanes { lanes, .. } => lanes * span,
        Traversal::Whole => {
            return Err(LowerError::StageUnsupported {
                why: "lane traversal required: one round is split across the workgroup's lanes",
            });
        }
    };
    let tiles = stage_b_tiles(st, lo, inner_axis, seg)?;
    // The per-lane segment base is computed before the round's bound is
    // tested: `round + lane·span` is what the guard compares with the extent,
    // so the compute may not sit inside the guard it feeds.
    let start = start.expect("lane traversal carries a start");
    Ok(Stmt::ForTiled {
        var: seg,
        axis: inner_axis,
        step,
        body: vec![Stmt::StageTiles {
            tiles,
            threads: st.threads,
            body: vec![
                Stmt::Compute(Inst {
                    dst: seg_body,
                    expr: LExpr::IAdd(seg, start),
                }),
                Stmt::InBounds {
                    bounds: vec![(seg_body, inner_axis)],
                    body,
                },
            ],
        }],
    })
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
    let rins: Vec<ValueId> = accs.iter().map(|&(_, _, rin, _)| rin).collect();
    dense_axis_reads(k, inner_axis, &rins)
}

/// The dense reads reachable from `rins` whose first index is the reduced
/// axis, whatever serves them: a run when the operand is contiguous, the
/// staged tile under `Schedule::stage_b`.
pub(crate) fn dense_axis_reads(k: &Kernel, inner_axis: AxisId, rins: &[ValueId]) -> Vec<ValueId> {
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
            let read_here = rins.iter().any(|rin| reaches(k, *rin, v));
            (dense && first_is_axis && rest_fixed && read_here).then_some(v)
        })
        .collect()
}

/// An accumulation whose block scale leaves the sum (`Schedule::factor_scales`):
/// the read it multiplies, what multiplies it, and the two partial sums one
/// segment accumulates before the scale is applied once.
#[derive(Clone, Copy)]
struct Factored {
    /// Index of the read among the segment's heads.
    head: usize,
    /// The other factor of the product.
    other: ValueId,
    /// `Σ q · other` over the segment.
    pq: VarId,
    /// `Σ other` over the segment, when the format has an offset to apply;
    /// shared by every row of a column when `other` does not depend on the
    /// row, and accumulated then by the first row alone.
    py: Option<(VarId, bool)>,
}

/// The read and the other factor of an accumulation `acc += dequant(a) · y`,
/// when it has that shape: a sum whose input is a product of one of the
/// segment's quantized reads and a value that does not read it.
fn factorable(k: &Kernel, heads: &[SegmentHeader], acc: &Accum) -> Option<(usize, ValueId)> {
    let (_, op, rin, _) = *acc;
    if op != rir_core::ReduceOp::Sum {
        return None;
    }
    let Op::Mul(x, y) = k.ops()[rin.0 as usize] else {
        return None;
    };
    heads.iter().enumerate().find_map(|(h, head)| {
        let q = head.read.value;
        if x == q && !reaches(k, y, q) {
            Some((h, y))
        } else if y == q && !reaches(k, x, q) {
            Some((h, x))
        } else {
            None
        }
    })
}

/// The element loop of a segment, in groups of `run` elements, for every point
/// of the tile.
///
/// ```text
/// [per point, factored: pq = 0; py = 0]
/// for g in 0..span/run:                         // constant trip count
///   col_c = seg + g·run + c                       for c in 0..run
///   per row: w = payload run of a at inb0+g·run   // one access
///   per column:
///     x_0..x_{run-1} = run of x at col_0          // one access, every row
///     per row: acc += f(w_c, x_c, …)   or   pq += q_c·x_c; py += x_c
/// [per point, factored: acc += s·pq − m·py]
/// ```
///
/// Each accumulator that is not factored still receives its terms in column
/// order, one per element, so its sum is the one the element loop computes;
/// what changes is that a dense operand is read once per run and per column
/// for every row, and the quantized payload once per run and per row when its
/// plan makes the run's bytes consecutive. With `run` one, every operand is
/// read element by element - the strided half of a run lowering, or a tile
/// with no run at all.
///
/// A factored accumulator receives one term per segment instead: the scale
/// and the offset of a block are the same for every element of a segment
/// that stays inside one sub-block, so `Σ (s·q − m)·x` is `s·Σ q·x − m·Σ x`,
/// summed in the segment's order and added once.
#[expect(clippy::too_many_arguments)]
fn lower_groups(
    lo: &mut Lowerer,
    heads: &[Vec<SegmentHeader>],
    runs: &[ValueId],
    seg: VarId,
    span: u32,
    run: u32,
    inner_axis: AxisId,
    tile: &PointTile,
    stage: Option<(&StageB, VarId)>,
) -> Result<Vec<Stmt>, LowerError> {
    let k = lo.k;
    let inner_name = k.axes()[inner_axis.0 as usize].name.clone();
    let (n_rows, n_cols) = (tile.n_rows(), tile.n_cols());
    let row_axis = tile.row_axis();
    let moves = |v: ValueId| row_axis.is_some_and(|ax| depends_on_axis(k, v, ax));

    // ---- once per segment: the scale and the offset each factored read
    // shares across the segment, and the partial sums, cleared.
    lo.begin_scope(true);
    let mut affine = Vec::with_capacity(n_rows);
    for row in heads {
        let mut per_head = Vec::with_capacity(row.len());
        for head in row {
            per_head.push(if lo.factor_scales {
                lo.lower_segment_affine(head.read.format, &head.header)?
            } else {
                None
            });
        }
        affine.push(per_head);
    }
    let mut factored: Vec<Vec<Vec<Option<Factored>>>> = vec![vec![Vec::new(); n_cols]; n_rows];
    let mut shared_py: HashMap<(usize, usize), VarId> = HashMap::new();
    for (r, per_row) in factored.iter_mut().enumerate() {
        for (c, per_point) in per_row.iter_mut().enumerate() {
            for (i, acc) in tile.accs[r][c].iter().enumerate() {
                let plan = factorable(k, &heads[r], acc)
                    .and_then(|(h, other)| affine[r][h].map(|aff| (h, other, aff)));
                let Some((head, other, aff)) = plan else {
                    per_point.push(None);
                    continue;
                };
                let pq = lo.new_var("pq", VarKind::F32);
                lo.body.push(Stmt::InitAcc {
                    acc: pq,
                    op: rir_core::ReduceOp::Sum,
                });
                let py = match aff.offset {
                    None => None,
                    Some(_) if !moves(other) => {
                        let owner = !shared_py.contains_key(&(c, i));
                        let py = *shared_py.entry((c, i)).or_insert_with(|| {
                            let py = lo.new_var("py", VarKind::F32);
                            lo.body.push(Stmt::InitAcc {
                                acc: py,
                                op: rir_core::ReduceOp::Sum,
                            });
                            py
                        });
                        Some((py, owner))
                    }
                    Some(_) => {
                        let py = lo.new_var("py", VarKind::F32);
                        lo.body.push(Stmt::InitAcc {
                            acc: py,
                            op: rir_core::ReduceOp::Sum,
                        });
                        Some((py, true))
                    }
                };
                per_point.push(Some(Factored {
                    head,
                    other,
                    pq,
                    py,
                }));
            }
        }
    }
    let mut out = lo.take_body();

    // ---- the group loop.
    let group = lo.new_var(&format!("{inner_name}_g"), VarKind::Idx);
    lo.begin_scope(true);
    let e0 = if run > 1 {
        lo.emit("e0", VarKind::Idx, LExpr::IMulC(group, run))
    } else {
        group
    };
    // The quantized payloads of the whole run in one access each, per row, when
    // the format's plans make the run's bytes consecutive (see
    // `lower_payload_run`); the decoded weights then do not depend on the
    // column and are shared by every column below.
    let mut payloads: Vec<Vec<Option<Vec<VarId>>>> = Vec::with_capacity(n_rows);
    for (r, row) in heads.iter().enumerate() {
        let mut per_head = Vec::with_capacity(row.len());
        for (h, head) in row.iter().enumerate() {
            if r > 0 && !moves(head.read.value) {
                per_head.push(payloads[0][h].clone());
                continue;
            }
            per_head.push(if run > 1 {
                let inb_first = lo.emit("inb0r", VarKind::Idx, LExpr::IAdd(head.inb0, e0));
                lo.lower_payload_run(head.read.arg, head.read.format, &head.base, inb_first, run)?
            } else {
                None
            });
        }
        payloads.push(per_head);
    }
    // Per element: its column, and per row and read its payload term and its
    // decoded weight. A factored accumulation reads the term, any other the
    // weight; the one nobody reads is pruned.
    let mut cols = Vec::with_capacity(run as usize);
    let mut terms: Vec<Vec<Vec<VarId>>> = vec![vec![Vec::new(); heads[0].len()]; n_rows];
    let mut values: Vec<Vec<Vec<VarId>>> = vec![vec![Vec::new(); heads[0].len()]; n_rows];
    for c in 0..run {
        let elem = if c == 0 {
            e0
        } else {
            lo.emit("e", VarKind::Idx, LExpr::IAddC(e0, c))
        };
        cols.push(lo.emit(&inner_name, VarKind::Idx, LExpr::IAdd(seg, elem)));
        for r in 0..n_rows {
            for (h, head) in heads[r].iter().enumerate() {
                if r > 0 && !moves(head.read.value) {
                    let (t, v) = (terms[0][h][c as usize], values[0][h][c as usize]);
                    terms[r][h].push(t);
                    values[r][h].push(v);
                    continue;
                }
                let inb = lo.emit("inb", VarKind::Idx, LExpr::IAdd(head.inb0, elem));
                let q = match &payloads[r][h] {
                    Some(qterms) => qterms[c as usize],
                    None => lo.lower_payload(head.read.arg, head.read.format, &head.base, inb)?,
                };
                let value = lo.lower_scale_apply(
                    head.read.arg,
                    head.read.format,
                    &head.base,
                    &head.header,
                    inb,
                    q,
                )?;
                terms[r][h].push(q);
                values[r][h].push(value);
            }
        }
    }
    // A strided dense operand still shares its scalar reads between rows.
    // Only the access width differs from the unit-stride branch.
    let scalar_reads;
    let cached_reads = if runs.is_empty() {
        scalar_reads = run_reads(k, inner_axis, &tile.accs[0][0]);
        scalar_reads.as_slice()
    } else {
        runs
    };
    let ctx = GroupCtx {
        inner_axis,
        row_axis,
        cols,
        heads,
        terms,
        values,
        runs: cached_reads,
        vector_reads: !runs.is_empty(),
        factored: &factored,
        stage: stage.map(|(st, tour)| StageCtx {
            round: st.round,
            tour,
            reads: st.reads.clone(),
        }),
    };
    let mut loaded: HashMap<(usize, usize, ValueId), Vec<VarId>> = HashMap::new();
    emit_points(lo, tile, &mut |lo, r, c, accs| {
        emit_group_point(lo, &ctx, &mut loaded, r, c, accs)
    })?;
    let body = lo.take_body();
    out.push(Stmt::ForConst {
        var: group,
        count: span / run,
        body,
    });

    // ---- once per segment: the factored sums, scaled and added.
    lo.begin_scope(true);
    for (r, per_row) in factored.iter().enumerate() {
        for (c, per_point) in per_row.iter().enumerate() {
            for (acc, f) in tile.accs[r][c].iter().zip(per_point) {
                let Some(f) = f else { continue };
                let aff = affine[r][f.head].expect("a factored read has an affine header");
                let scaled = lo.emit("t", VarKind::F32, LExpr::Mul(aff.scale, f.pq));
                let term = match (aff.offset, f.py) {
                    (Some(off), Some((py, _))) => {
                        let shift = lo.emit("t", VarKind::F32, LExpr::Mul(off.value, py));
                        let expr = if off.subtract {
                            LExpr::Sub(scaled, shift)
                        } else {
                            LExpr::Add(scaled, shift)
                        };
                        lo.emit("t", VarKind::F32, expr)
                    }
                    _ => scaled,
                };
                lo.body.push(Stmt::Accum {
                    acc: acc.0,
                    op: acc.1,
                    value: term,
                });
            }
        }
    }
    out.extend(lo.take_body());
    Ok(out)
}

/// What one group loop resolved before its accumulations: the column of every
/// element, the payload term and the decoded weight of every element per row
/// and read, the dense reads to serve in runs, and the factored accumulations.
struct GroupCtx<'a> {
    inner_axis: AxisId,
    row_axis: Option<AxisId>,
    cols: Vec<VarId>,
    heads: &'a [Vec<SegmentHeader>],
    /// `terms[r][h][c]`: the payload term the scale multiplies.
    terms: Vec<Vec<Vec<VarId>>>,
    /// `values[r][h][c]`: the decoded weight.
    values: Vec<Vec<Vec<VarId>>>,
    runs: &'a [ValueId],
    vector_reads: bool,
    factored: &'a [Vec<Vec<Option<Factored>>>],
    /// The staged shared operand, when `Schedule::stage_b` is on: its reads are
    /// served from the tile instead of the buffer.
    stage: Option<StageCtx>,
}

/// The staged tile as the group loop reads it: the round's width (the tile's
/// row stride), the round loop's variable, and the reads with their arrays.
struct StageCtx {
    round: u32,
    tour: VarId,
    reads: Vec<super::stage::StagedRead>,
}

impl StageCtx {
    /// The tile array serving `rv`, when the read is staged.
    fn tile_of(&self, rv: ValueId) -> Option<VarId> {
        self.reads.iter().find(|r| r.value == rv).map(|r| r.tile)
    }

    /// Reads `rv` from the staged tile at column `col` of the round `tour`:
    /// the tile's element `(point, col - tour)` is `point · round + col - tour`.
    fn load(
        &self,
        lo: &mut Lowerer,
        rv: ValueId,
        point: usize,
        col: VarId,
    ) -> Result<VarId, LowerError> {
        let tile = self.tile_of(rv).expect("load is called for a staged read");
        let off = lo.emit("boff", VarKind::Idx, LExpr::ISub(col, self.tour));
        let at = lo.emit(
            "bat",
            VarKind::Idx,
            LExpr::IAddC(off, point as u32 * self.round),
        );
        let dst = lo.new_var("t", VarKind::F32);
        lo.body.push(Stmt::LoadShared {
            dst,
            array: tile,
            index: at,
            width: 1,
        });
        Ok(dst)
    }
}

/// One point's accumulations over a group: the runs read at the group's first
/// column - once per column, and per row only when the run reads the row -
/// then the level once per element, each seeing its own column, weight and run
/// element.
///
/// Every value that depends on the reduced axis is dropped from the memo
/// between elements, or element `c` would reuse a register computed at `c - 1`.
fn emit_group_point(
    lo: &mut Lowerer,
    ctx: &GroupCtx,
    loaded: &mut HashMap<(usize, usize, ValueId), Vec<VarId>>,
    r: usize,
    c: usize,
    accs: &[Accum],
) -> Result<(), LowerError> {
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
        let row_key = if ctx.row_axis.is_some_and(|ax| depends_on_axis(k, rv, ax)) {
            r
        } else {
            0
        };
        if let Some(regs) = loaded.get(&(c, row_key, rv)) {
            run_regs.push(regs.clone());
            continue;
        }
        if !ctx.vector_reads {
            let mut regs = Vec::with_capacity(ctx.cols.len());
            for &col in &ctx.cols {
                lo.env = keep(&lo.env);
                lo.axis_vars.insert(inner, col);
                // A staged read comes from the tile, not from the buffer.
                let reg = match &ctx.stage {
                    Some(st) if st.tile_of(rv).is_some() => st.load(lo, rv, c, col)?,
                    _ => lo.lower_value(rv)?,
                };
                regs.push(reg);
            }
            lo.env = keep(&lo.env);
            loaded.insert((c, row_key, rv), regs.clone());
            run_regs.push(regs);
            continue;
        }
        // A staged read is served from the tile: one shared load per element
        // of the run, at the column the element walks.
        if let Some(st) = &ctx.stage
            && st.tile_of(rv).is_some()
        {
            let mut regs = Vec::with_capacity(ctx.cols.len());
            for &col in &ctx.cols {
                regs.push(st.load(lo, rv, c, col)?);
            }
            loaded.insert((c, row_key, rv), regs.clone());
            run_regs.push(regs);
            continue;
        }
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
        loaded.insert((c, row_key, rv), dsts.clone());
        run_regs.push(dsts);
    }
    let base = keep(&lo.env);
    for (e, &col) in ctx.cols.iter().enumerate() {
        lo.env = base.clone();
        lo.axis_vars.insert(inner, col);
        for (h, head) in ctx.heads[r].iter().enumerate() {
            lo.env.insert(head.read.value, ctx.values[r][h][e]);
        }
        for (v, regs) in ctx.runs.iter().zip(&run_regs) {
            lo.env.insert(*v, regs[e]);
        }
        for (acc, f) in accs.iter().zip(&ctx.factored[r][c]) {
            let Some(f) = f else {
                emit_accums(lo, std::slice::from_ref(acc))?;
                continue;
            };
            let other = lo.lower_value(f.other)?;
            let term = lo.emit(
                "t",
                VarKind::F32,
                LExpr::Mul(ctx.terms[r][f.head][e], other),
            );
            lo.body.push(Stmt::Accum {
                acc: f.pq,
                op: rir_core::ReduceOp::Sum,
                value: term,
            });
            if let Some((py, true)) = f.py {
                lo.body.push(Stmt::Accum {
                    acc: py,
                    op: rir_core::ReduceOp::Sum,
                    value: other,
                });
            }
        }
    }
    lo.env = keep(&lo.env);
    Ok(())
}

/// The accumulations of one point, as `emit_points` asks for them: the point's
/// row, its column, and its accumulators.
type EmitPoint<'f> = dyn FnMut(&mut Lowerer, usize, usize, &[Accum]) -> Result<(), LowerError> + 'f;

/// The accumulations of one level once per point of the tile.
///
/// The first column is lowered unguarded, row after row, and what each row
/// computed is kept for the points that can read it: what depends on neither
/// axis for every point, what depends on the row alone - the decoded weight,
/// the scale of its block - for that row's other columns, and what depends on
/// the column alone - the other operand - for that column's other rows. That
/// is the whole point: one read and one decode per row for the columns, and
/// one read of the other operand per column for the rows. Each further column
/// is guarded (`Stmt::InBounds`), since the last tile of an axis may be partial
/// and its reads must not leave the tensor; the first needs no guard because
/// the grid already returned past the axis, and no row does, because each row
/// reads a register wrapped into the axis.
fn emit_points(
    lo: &mut Lowerer,
    tile: &PointTile,
    emit: &mut EmitPoint<'_>,
) -> Result<(), LowerError> {
    let k = lo.k;
    let (row_axis, col_axis) = (tile.row_axis(), tile.col_axis());
    let on = |v: &ValueId, ax: Option<AxisId>| ax.is_some_and(|a| depends_on_axis(k, *v, a));
    let merged = |sets: [&HashMap<ValueId, VarId>; 3]| -> HashMap<ValueId, VarId> {
        sets.into_iter()
            .flat_map(|s| s.iter().map(|(v, r)| (*v, *r)))
            .collect()
    };
    let mut base: HashMap<ValueId, VarId> = lo
        .env
        .iter()
        .filter(|(v, _)| !on(v, row_axis) && !on(v, col_axis))
        .map(|(v, r)| (*v, *r))
        .collect();
    let mut by_row: Vec<HashMap<ValueId, VarId>> = vec![HashMap::new(); tile.n_rows()];

    let mut by_col: HashMap<ValueId, VarId> = HashMap::new();
    for (r, row_env) in by_row.iter_mut().enumerate() {
        lo.env = merged([&base, row_env, &by_col]);
        tile.bind(lo, r, 0);
        emit(lo, r, 0, &tile.accs[r][0])?;
        for (v, reg) in &lo.env {
            match (on(v, row_axis), on(v, col_axis)) {
                (false, false) => {
                    base.insert(*v, *reg);
                }
                (true, false) => {
                    row_env.insert(*v, *reg);
                }
                (false, true) => {
                    by_col.insert(*v, *reg);
                }
                (true, true) => {}
            }
        }
    }
    for c in 1..tile.n_cols() {
        let (axis, regs) = tile
            .cols
            .expect("a tile with several columns has a column axis");
        let outer = lo.take_body();
        let mut by_col: HashMap<ValueId, VarId> = HashMap::new();
        for (r, row_env) in by_row.iter().enumerate() {
            lo.env = merged([&base, row_env, &by_col]);
            tile.bind(lo, r, c);
            emit(lo, r, c, &tile.accs[r][c])?;
            for (v, reg) in &lo.env {
                if on(v, col_axis) && !on(v, row_axis) {
                    by_col.insert(*v, *reg);
                }
            }
        }
        let guarded = lo.take_body();
        lo.body = outer;
        lo.body.push(Stmt::InBounds {
            bounds: vec![(regs[c], axis)],
            body: guarded,
        });
    }
    tile.bind(lo, 0, 0);
    lo.env = base;
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
        Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) | Op::Erf(a) => collect_result_deps(k, *a, out),
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
    rows: Option<RowTile>,
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

    // Under a row tile, the subgroup's index when the workgroup holds several,
    // then one register per row the subgroup owns and the one its reads use:
    // the same index wrapped into the axis, so a row past the extent reads a
    // row that exists and only its write is held back. The grid's own index
    // is in range by construction and is read as it is.
    let sg_var = rows
        .filter(|t| t.subgroups > 1)
        .map(|_| lo.new_var("sg", VarKind::Idx));
    let mut row_idx: Vec<VarId> = Vec::new();
    let mut row_reads: Vec<VarId> = Vec::new();
    if let Some(t) = rows {
        let axis_name = lo.k.axes()[t.axis.0 as usize].name.clone();
        let first = match sg_var {
            Some(sg) => {
                let off = lo.new_var(&format!("{axis_name}_sg"), VarKind::Idx);
                lane_body.push(Stmt::Compute(Inst {
                    dst: off,
                    expr: LExpr::IMulC(sg, t.per_subgroup),
                }));
                let first = lo.new_var(&format!("{axis_name}_first"), VarKind::Idx);
                lane_body.push(Stmt::Compute(Inst {
                    dst: first,
                    expr: LExpr::IAdd(t.first, off),
                }));
                first
            }
            None => t.first,
        };
        for r in 0..t.per_subgroup {
            let idx = if r == 0 {
                first
            } else {
                let v = lo.new_var(&format!("{axis_name}{r}"), VarKind::Idx);
                lane_body.push(Stmt::Compute(Inst {
                    dst: v,
                    expr: LExpr::IAddC(first, r),
                }));
                v
            };
            let read = if sg_var.is_none() && r == 0 {
                idx
            } else {
                let v = lo.new_var(&format!("{axis_name}_rd"), VarKind::Idx);
                lane_body.push(Stmt::Compute(Inst {
                    dst: v,
                    expr: LExpr::IModAxis {
                        var: idx,
                        axis: t.axis,
                    },
                }));
                v
            };
            row_idx.push(idx);
            row_reads.push(read);
        }
    }
    // Whether the write of row `r` needs its bound: every row but the grid's
    // own index.
    let row_guarded = |r: usize| rows.is_some() && (sg_var.is_some() || r > 0);
    let n_rows = row_idx.len().max(1);
    let n_cols = tile_idx.len().max(1);
    let mut results: Vec<Vec<HashMap<ValueId, VarId>>> =
        vec![vec![lo.results_env.clone(); n_cols]; n_rows];

    if !a.reduces.is_empty() {
        let plan = plan_segments(lo.k, inner_axis, lo.segment)?;
        // A quantized read whose position moves with the tiled axis would have
        // its block header decoded once, at the first index, and reused at the
        // others. Nothing tiled reads one; refusing it is what keeps that true.
        // The row tile decodes one header per row, so it has no such rule.
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
        // The staged shared operand: one tile of `lane_tile` columns by
        // `lanes · span` reduced elements per round, loaded once by the whole
        // workgroup and read by every subgroup for its own rows. One array per
        // dense read along the reduced axis, declared in `LoopKernel::shared`.
        if lo.stage_b {
            let Some(t) = tile else {
                return Err(LowerError::StageUnsupported {
                    why: "a lane tile is required: the staged tile is one column per tile index",
                });
            };
            let rins: Vec<ValueId> = a.reduces.iter().map(|&(_, _, rin)| rin).collect();
            // A tile is shared by every row and subgroup in the workgroup.
            // Row-dependent reads must keep their per-row global loads: using
            // a row register in the cooperative loader would mix different
            // rows in one shared array.
            let reads: Vec<_> = dense_axis_reads(lo.k, inner_axis, &rins)
                .into_iter()
                .filter(|&rv| rows.is_none_or(|r| !depends_on_axis(lo.k, rv, r.axis)))
                .collect();
            if reads.is_empty() {
                return Err(LowerError::StageUnsupported {
                    why: "no row-independent dense operand is read along the reduced axis: nothing to stage",
                });
            }
            let threads = lanes * rows.map_or(1, |t| t.subgroups);
            let mut staged = Vec::with_capacity(reads.len());
            for rv in reads {
                let tile_var = lo.new_var("tile_b", VarKind::Idx);
                lo.shared.push((tile_var, lanes * plan.span * t.count));
                staged.push(StagedRead {
                    value: rv,
                    tile: tile_var,
                });
            }
            lo.stage = Some(StageB {
                threads,
                round: lanes * plan.span,
                cols: t.count,
                cols_first: t.first,
                col_axis: t.axis,
                reads: staged,
            });
        }
        // Use the same dependency **levels** as the sequential path. Each
        // level is a strided loop followed by its collectives; because
        // `subgroupAdd` broadcasts, the next level sees each result in every
        // lane.
        for group in reduce_levels(lo.k, &a.reduces).values() {
            // The first point's accumulators, then every other point's, row by
            // row - the order the registers have always been numbered in.
            let mut accs: Vec<Vec<Vec<Accum>>> = vec![vec![Vec::new(); n_cols]; n_rows];
            for (r, per_row) in accs.iter_mut().enumerate() {
                for (c, per_point) in per_row.iter_mut().enumerate() {
                    for (rv, rop, rin) in group {
                        let name = match (r, c, rows) {
                            (0, 0, _) => format!("acc{}", rv.0),
                            (_, _, None) => format!("acc{}_{c}", rv.0),
                            (_, _, Some(_)) => format!("acc{}_{r}_{c}", rv.0),
                        };
                        let acc = lo.new_var(&name, VarKind::F32);
                        lane_body.push(Stmt::InitAcc { acc, op: *rop });
                        per_point.push((acc, *rop, *rin, *rv));
                    }
                }
            }
            let points = PointTile {
                rows: rows.map(|t| (t.axis, row_reads.as_slice())),
                cols: tile.map(|t| (t.axis, tile_idx.as_slice())),
                accs: &accs,
                results: &results,
            };

            // Results from earlier levels are visible. Referring to a level
            // not yet computed yields `NestedReduce` through an environment
            // lookup miss.
            lane_body.extend(lower_accum_loop(
                lo,
                &plan,
                inner_axis,
                &accs[0][0],
                Traversal::Lanes {
                    lane: lane_var,
                    lanes,
                },
                Some(&points),
            )?);

            // The collectives of one level, together. `SubgroupTree` reduces
            // through a subgroup instruction and has no barrier to share, so it
            // keeps one statement per accumulator; `SharedTree` expands a tree
            // in shared memory, and grouping is what makes its levels - and its
            // barriers - serve every accumulator at once.
            let mut reds = Vec::new();
            for r in 0..n_rows {
                for c in 0..n_cols {
                    for (acc, rop, _, rv) in &accs[r][c] {
                        let red = lo.new_var(&format!("red{}", rv.0), VarKind::F32);
                        match strategy {
                            ReductionStrategy::SubgroupTree => lane_body.push(Stmt::LaneReduce {
                                op: *rop,
                                src: *acc,
                                dst: red,
                            }),
                            ReductionStrategy::SharedTree | ReductionStrategy::HierarchicalTree => {
                                // One array per accumulator, even grouped: what
                                // the group shares is the barriers, not the
                                // storage.
                                //
                                // Its length is the lane count for the flat
                                // tree, whose first level reads every lane's
                                // slot, and the *subgroup count* for the
                                // hierarchical one, which never stores more
                                // than one total per subgroup. 256 lanes are
                                // 1 KiB of threadgroup storage per accumulator
                                // against 32 bytes.
                                lo.shared.push((red, subgroup.map_or(lanes, |w| lanes / w)));
                                reds.push(GroupRed {
                                    op: *rop,
                                    src: *acc,
                                    dst: red,
                                })
                            }
                            _ => unreachable!("lower_lanes is called only for a lane strategy"),
                        }
                        results[r][c].insert(*rv, red);
                    }
                }
            }
            if n_rows * n_cols == 1 {
                lo.results_env = results[0][0].clone();
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
        let stmts = if n_rows * n_cols == 1 {
            lower_writes(lo, &a.row_writes)?
        } else {
            // One write per point, each reading its own results, and every
            // point but the grid's own guarded on the axes it moved along.
            let mut stmts = Vec::new();
            for r in 0..n_rows {
                for c in 0..n_cols {
                    let mut bounds = Vec::new();
                    if let Some(t) = rows {
                        lo.axis_vars.insert(t.axis, row_idx[r]);
                        if row_guarded(r) {
                            bounds.push((row_idx[r], t.axis));
                        }
                    }
                    if let Some(t) = tile {
                        lo.axis_vars.insert(t.axis, tile_idx[c]);
                        if c > 0 {
                            bounds.push((tile_idx[c], t.axis));
                        }
                    }
                    lo.results_env = results[r][c].clone();
                    let writes = lower_writes(lo, &a.row_writes)?;
                    if bounds.is_empty() {
                        stmts.extend(writes);
                    } else {
                        stmts.push(Stmt::InBounds {
                            bounds,
                            body: writes,
                        });
                    }
                }
            }
            stmts
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

    let lane = Stmt::ParallelLane {
        var: lane_var,
        lanes,
        body: lane_body,
    };
    Ok(vec![match (sg_var, rows) {
        (Some(var), Some(t)) => Stmt::ParallelSubgroup {
            var,
            count: t.subgroups,
            body: vec![lane],
        },
        _ => lane,
    }])
}
