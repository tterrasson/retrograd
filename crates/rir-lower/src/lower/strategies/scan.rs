//! Scans: the blocked scan across lanes, the tiled scan and the shared
//! Blelloch tree it lowers to.

use rir_core::ScanDirection;

use crate::loop_ir::*;
use crate::schedule::ReductionStrategy;

use crate::lower::{Analysis, LowerError, Lowerer, lower_writes};

/// Blocked scan (`ScanStrategy::BlockedLanes`): three phases inside one
/// workgroup, no shared memory and no barrier - the lane collective carries
/// everything that crosses lanes.
///
/// 1. each lane totals its contiguous chunk;
/// 2. an exclusive lane prefix turns those totals into per-lane offsets;
/// 3. each lane re-traverses its chunk, accumulating from its offset, and
///    writes each partial result.
///
/// Phase 3 re-reads the input rather than keeping phase 1's values: a register
/// per element would bound the row length to the register file. Two reads for
/// a serial depth divided by the lane count is the trade this strategy makes.
pub(crate) fn lower_blocked_scan(
    lo: &mut Lowerer,
    a: &Analysis,
    lanes: u32,
    strategy: ReductionStrategy,
) -> Result<Vec<Stmt>, LowerError> {
    let inner_axis = a.inner_axis.expect("blocked scan: scan axis required");
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let lane_var = lo.new_var("lane", VarKind::Idx);
    let reverse = a.scans[0].2 == ScanDirection::Backward;
    let mut lane_body = Vec::new();

    // Phase 1 - total of the lane's block.
    let mut parts = Vec::new();
    for (sv, sop, _, sin) in &a.scans {
        let part = lo.new_var(&format!("part{}", sv.0), VarKind::F32);
        let rop = match sop {
            rir_core::ScanOp::Sum => rir_core::ReduceOp::Sum,
        };
        lane_body.push(Stmt::InitAcc { acc: part, op: rop });
        parts.push((part, rop, *sin, *sv));
    }
    let col1 = lo.new_var(&inner_name, VarKind::Idx);
    lo.axis_vars.insert(inner_axis, col1);
    lo.begin_scope(true);
    for (part, rop, sin, _) in &parts {
        let v = lo.lower_value(*sin)?;
        lo.body.push(Stmt::Accum {
            acc: *part,
            op: *rop,
            value: v,
        });
    }
    let phase1 = lo.take_body();
    lane_body.push(Stmt::ForChunk {
        var: col1,
        axis: inner_axis,
        lane: lane_var,
        lanes,
        reverse,
        body: phase1,
    });

    // Phase 2 - exclusive prefix of totals: the lane's starting offset.
    let mut states = Vec::new();
    for (part, rop, sin, sv) in &parts {
        let off = lo.new_var(&format!("off{}", sv.0), VarKind::F32);
        match strategy {
            // A subgroup prefix is a **primitive**: one instruction the backend
            // offers, which is exactly what an emitter may still print.
            ReductionStrategy::SubgroupTree => lane_body.push(Stmt::LaneScan {
                op: *rop,
                src: *part,
                dst: off,
            }),
            // The workgroup-wide one is an algorithm, so it is lowered here.
            // One array per scanned value: the tree
            // is live across barriers, so two scans cannot share slots.
            ReductionStrategy::SharedTree => {
                let sh = lo.new_var(&format!("scan_sh{}", sv.0), VarKind::F32);
                lo.shared.push((sh, lanes));
                let stmts = workgroup_scan_stmts(lo, *rop, lanes, sh, lane_var, *part, off);
                lane_body.extend(stmts);
            }
            _ => unreachable!("lower_blocked_scan is called only for a lane strategy"),
        }
        states.push((off, *rop, *sin, *sv));
    }

    // Phase 3 - traverse the block again from the offset. The accumulator is
    // initialized to the identity and then combined with the offset rather than
    // copied: this holds for any scan operator, not only sum.
    let col2 = lo.new_var(&inner_name, VarKind::Idx);
    let mut accs = Vec::new();
    for (off, rop, sin, sv) in &states {
        let acc = lo.new_var(&format!("scan{}", sv.0), VarKind::F32);
        lane_body.push(Stmt::InitAcc { acc, op: *rop });
        lane_body.push(Stmt::Accum {
            acc,
            op: *rop,
            value: *off,
        });
        accs.push((acc, *rop, *sin, *sv));
    }
    lo.axis_vars.insert(inner_axis, col2);
    lo.begin_scope(true);
    for (acc, rop, sin, sv) in &accs {
        let v = lo.lower_value(*sin)?;
        lo.body.push(Stmt::Accum {
            acc: *acc,
            op: *rop,
            value: v,
        });
        let val = lo.new_var(&format!("s{}", sv.0), VarKind::F32);
        lo.body.push(Stmt::Compute(Inst {
            dst: val,
            expr: LExpr::Copy(*acc),
        }));
        lo.results_env.insert(*sv, val);
    }
    for (tensor, idx, value) in &a.inner_writes {
        lo.lower_store(*tensor, idx, *value)?;
    }
    let phase3 = lo.take_body();
    lane_body.push(Stmt::ForChunk {
        var: col2,
        axis: inner_axis,
        lane: lane_var,
        lanes,
        reverse,
        body: phase3,
    });

    if !a.row_writes.is_empty() {
        lo.axis_vars.remove(&inner_axis);
        let stmts = lower_writes(lo, &a.row_writes)?;
        lane_body.push(Stmt::LaneZero {
            lane: lane_var,
            body: stmts,
        });
    }

    Ok(vec![Stmt::ParallelLane {
        var: lane_var,
        lanes,
        body: lane_body,
    }])
}

/// Coalesced tiled scan (`ScanStrategy::TiledLanes`): the workgroup walks the
/// scanned axis in tiles of `lanes · items`, and everything that crosses lanes
/// happens in shared memory (ADR-2 section 5).
///
/// The whole strategy is lowered here - the tile loop, interleaved staging,
/// barriers, per-lane serial scan, and prefix over lane totals. What varies from kernel
/// to kernel stays what it was: the two **bodies**, `load` and `store`, written
/// in terms of one axis register, which are the part that knows a view's
/// strides, a fused decoder, an element type.
///
/// The restrictions are narrow and stated rather than worked around: one scan,
/// no reduction beside it, and writes that depend on the scanned axis. A kernel
/// outside that shape gets `TiledScanUnsupported`, not a quiet blocked scan.
pub(crate) fn lower_tiled_scan(
    lo: &mut Lowerer,
    a: &Analysis,
    lanes: u32,
    items: u32,
) -> Result<Vec<Stmt>, LowerError> {
    let unsupported = |why: &'static str| LowerError::TiledScanUnsupported { why };

    if items == 0 {
        return Err(unsupported("zero items: an empty tile scans nothing"));
    }
    if a.scans.len() != 1 {
        return Err(unsupported(
            "only one scan: the tile carries one value per element, not several",
        ));
    }
    if !a.reduces.is_empty() {
        return Err(unsupported(
            "reduction in the same kernel: both collectives would contend for storage",
        ));
    }
    if !a.row_writes.is_empty() {
        return Err(unsupported(
            "write independent of the scanned axis: nothing says which tile it belongs to",
        ));
    }
    if a.inner_writes.is_empty() {
        return Err(unsupported("no write depends on the scanned axis"));
    }

    let (sv, sop, dir, sin) = a.scans[0];
    let op = match sop {
        rir_core::ScanOp::Sum => rir_core::ReduceOp::Sum,
    };
    let reverse = dir == ScanDirection::Backward;
    let inner_axis = a.inner_axis.expect("tiled scan: scan axis required");
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let width = lanes * items;

    let lane_var = lo.new_var("lane", VarKind::Idx);
    let tile = lo.new_var("scan_tile", VarKind::Idx);
    // The tile is one word longer per lane than it holds elements, and that
    // padding is the access plan of the middle phase: a lane's run has to be
    // contiguous *in sequence order*, so lane `l` reads element `l · items + j`
    // - a stride of `items` across lanes, which lands every lane in the same
    // memory bank as soon as `items` is a multiple of the bank count. Storing
    // element `q` at `q + q / items` makes that stride `items + 1`, coprime
    // with any power-of-two bank count, and leaves the two coalesced phases
    // walking consecutive words (ADR-2 section 5).
    lo.shared.push((tile, width + lanes));
    let totals = lo.new_var("scan_totals", VarKind::F32);
    lo.shared.push((totals, lanes));
    let base = lo.new_var(&format!("{inner_name}_base"), VarKind::Idx);
    let global = lo.new_var(&inner_name, VarKind::Idx);
    let value = lo.new_var(&format!("elem{}", sv.0), VarKind::F32);
    // The running total across tiles: declared once outside the tile loop and
    // **replaced** at the end of every round, which is the one loop-carried
    // value of the algorithm and the reason `Stmt::Set` exists.
    let carry = lo.new_var("carry", VarKind::F32);

    // `load`: the input element at `global`, left in `value`. The copy is not
    // redundant: the next phase requires the element in
    // `value`, and what `lower_value` returns is whichever register happened to
    // hold it.
    lo.axis_vars.insert(inner_axis, global);
    lo.begin_scope(true);
    let v = lo.lower_value(sin)?;
    lo.body.push(Stmt::Compute(Inst {
        dst: value,
        expr: LExpr::Copy(v),
    }));
    let load = lo.take_body();

    // `store`: the scanned result, read back from the tile into `value`, then
    // written where the kernel says. Binding the scan's own value to `value` is
    // what lets an arbitrary write expression sit downstream of the scan.
    lo.begin_scope(true);
    lo.results_env.insert(sv, value);
    for (tensor, idx, val) in &a.inner_writes {
        lo.lower_store(*tensor, idx, *val)?;
    }
    let store = lo.take_body();
    lo.axis_vars.remove(&inner_axis);

    // The axis index of tile-local slot `slot`: `base + slot`, mirrored
    // end-to-start for a backward scan so one traversal serves both directions.
    // Written as a closure because the two cooperative phases each need it, in
    // their own scope - the same code, not a shared register.
    let axis_index = |lo: &mut Lowerer, out: &mut Vec<Stmt>, p: VarId| {
        let expr = if reverse {
            let n = push_expr(lo, out, "n", VarKind::Idx, LExpr::AxisExtentIdx(inner_axis));
            let last = push_expr(lo, out, "last", VarKind::Idx, LExpr::ISubC(n, 1));
            LExpr::ISub(last, p)
        } else {
            LExpr::Copy(p)
        };
        out.push(Stmt::Compute(Inst { dst: global, expr }));
    };

    // Slot arithmetic, identical in both cooperative phases: lane `l` owns
    // slots `l`, `l + lanes`, … so consecutive lanes read consecutive
    // addresses, and the padded position of a slot is `slot + slot / items`.
    let slot_of = |lo: &mut Lowerer, out: &mut Vec<Stmt>, i: VarId| -> (VarId, VarId, VarId) {
        let scaled = push_expr(lo, out, "s0", VarKind::Idx, LExpr::IMulC(i, lanes));
        let slot = push_expr(lo, out, "slot", VarKind::Idx, LExpr::IAdd(scaled, lane_var));
        let p = push_expr(lo, out, "p", VarKind::Idx, LExpr::IAdd(base, slot));
        let pad = push_expr(lo, out, "pad", VarKind::Idx, LExpr::IDivC(slot, items));
        let at = push_expr(lo, out, "at", VarKind::Idx, LExpr::IAdd(slot, pad));
        (slot, p, at)
    };

    let mut round: Vec<Stmt> = Vec::new();

    // Phase 1 - stage the tile, interleaved. A slot past the end of the axis
    // keeps the identity, which is what makes the last tile of a row and the
    // running total below both correct without a second test.
    round.push(Stmt::Barrier);
    let i = lo.new_var("i", VarKind::Idx);
    let mut fill: Vec<Stmt> = Vec::new();
    let (_, p, at) = slot_of(lo, &mut fill, i);
    let zero = lo.new_var("zero", VarKind::F32);
    fill.push(Stmt::InitAcc { acc: zero, op });
    fill.push(Stmt::StoreShared {
        array: tile,
        index: at,
        value: zero,
    });
    let mut inside = Vec::new();
    axis_index(lo, &mut inside, p);
    inside.extend(load);
    inside.push(Stmt::StoreShared {
        array: tile,
        index: at,
        value,
    });
    fill.push(Stmt::InBounds {
        bounds: vec![(p, inner_axis)],
        body: inside,
    });
    round.push(Stmt::ForConst {
        var: i,
        count: items,
        body: fill,
    });
    round.push(Stmt::Barrier);

    // Phase 2 - each lane totals its own run, an exclusive prefix over those
    // totals gives it its offset, and it re-scans its run from there. The
    // accumulator starts at the identity and is combined with the carry and the
    // offset rather than copied, so this holds for any scan operator.
    let run = push_expr(
        lo,
        &mut round,
        "run",
        VarKind::Idx,
        LExpr::IMulC(lane_var, items + 1),
    );
    let part = lo.new_var("part", VarKind::F32);
    round.push(Stmt::InitAcc { acc: part, op });
    let j = lo.new_var("j", VarKind::Idx);
    let mut total = Vec::new();
    let at1 = push_expr(lo, &mut total, "at", VarKind::Idx, LExpr::IAdd(run, j));
    let t1 = lo.new_var("t", VarKind::F32);
    total.push(Stmt::LoadShared {
        dst: t1,
        array: tile,
        index: at1,
        width: 1,
    });
    total.push(Stmt::Accum {
        acc: part,
        op,
        value: t1,
    });
    round.push(Stmt::ForConst {
        var: j,
        count: items,
        body: total,
    });

    let off = lo.new_var("off", VarKind::F32);
    round.extend(workgroup_scan_stmts(
        lo, op, lanes, totals, lane_var, part, off,
    ));

    let acc = lo.new_var("acc", VarKind::F32);
    round.push(Stmt::InitAcc { acc, op });
    round.push(Stmt::Accum {
        acc,
        op,
        value: carry,
    });
    round.push(Stmt::Accum {
        acc,
        op,
        value: off,
    });
    let j2 = lo.new_var("j", VarKind::Idx);
    let mut scan = Vec::new();
    let at2 = push_expr(lo, &mut scan, "at", VarKind::Idx, LExpr::IAdd(run, j2));
    let t2 = lo.new_var("t", VarKind::F32);
    scan.push(Stmt::LoadShared {
        dst: t2,
        array: tile,
        index: at2,
        width: 1,
    });
    scan.push(Stmt::Accum { acc, op, value: t2 });
    scan.push(Stmt::StoreShared {
        array: tile,
        index: at2,
        value: acc,
    });
    round.push(Stmt::ForConst {
        var: j2,
        count: items,
        body: scan,
    });
    round.push(Stmt::Barrier);

    // The running total is read off the tile's last slot, not recombined from
    // the lane totals: out-of-range slots hold the identity, so that slot *is*
    // the scan of everything seen so far - and it is the one value every lane
    // can read without a second collective.
    let last = push_expr(
        lo,
        &mut round,
        "last",
        VarKind::Idx,
        LExpr::ConstIdx((width - 1) + (width - 1) / items),
    );
    let tail = lo.new_var("tail", VarKind::F32);
    round.push(Stmt::LoadShared {
        dst: tail,
        array: tile,
        index: last,
        width: 1,
    });
    round.push(Stmt::Set {
        var: carry,
        value: tail,
    });

    // Phase 3 - flush the tile the same interleaved way it was staged.
    let i2 = lo.new_var("i", VarKind::Idx);
    let mut flush: Vec<Stmt> = Vec::new();
    let (_, p2, at3) = slot_of(lo, &mut flush, i2);
    let mut out_body = Vec::new();
    axis_index(lo, &mut out_body, p2);
    out_body.push(Stmt::LoadShared {
        dst: value,
        array: tile,
        index: at3,
        width: 1,
    });
    out_body.extend(store);
    flush.push(Stmt::InBounds {
        bounds: vec![(p2, inner_axis)],
        body: out_body,
    });
    round.push(Stmt::ForConst {
        var: i2,
        count: items,
        body: flush,
    });

    let mut lane_body = vec![Stmt::InitAcc { acc: carry, op }];
    lane_body.push(Stmt::ForTiled {
        var: base,
        axis: inner_axis,
        step: width,
        body: round,
    });

    Ok(vec![Stmt::ParallelLane {
        var: lane_var,
        lanes,
        body: lane_body,
    }])
}

/// One `Compute` into a statement list, and the register it defines.
///
/// `Lowerer::emit` writes to the current scope's buffer; the collectives below
/// build their own lists - a tree level, a cooperative phase - so they need the
/// same one-liner against an explicit list.
pub(crate) fn push_expr(
    lo: &mut Lowerer,
    out: &mut Vec<Stmt>,
    base: &str,
    kind: VarKind,
    expr: LExpr,
) -> VarId {
    let dst = lo.new_var(base, kind);
    out.push(Stmt::Compute(Inst { dst, expr }));
    dst
}

/// The workgroup-wide exclusive prefix, **lowered**.
///
/// This is the Blelloch tree: an upsweep of `log2(lanes)` levels, the last slot
/// cleared, a
/// downsweep of as many, and a barrier after every one of them. Nothing about
/// it is syntax - it is a series of levels and a plan of barriers, which is the
/// definition this crate gives of an algorithm, and the reason it may not sit in an
/// emitter. `interp` executes exactly what the two backends print.
///
/// The levels are **unrolled** rather than written as a doubling loop, and that
/// is not a transformation: `lanes` is fixed by the schedule, so `2^l` is a
/// constant of the same kind as a `ForConst` trip count. What the unrolling
/// costs is print size; what it saves is a loop form in the IR whose only user
/// would be this function.
///
/// The topology is unchanged, level for level: `sh[idx] = combine(sh[idx - s],
/// sh[idx])` on the way up with `idx = lane · 2s + (2s - 1)`, and the swap on
/// the way down. `sh` must be a declared shared array of `lanes` elements, and
/// `dst` a register nothing else defines: it receives lane `l`'s prefix.
pub(crate) fn workgroup_scan_stmts(
    lo: &mut Lowerer,
    op: rir_core::ReduceOp,
    lanes: u32,
    sh: VarId,
    lane: VarId,
    src: VarId,
    dst: VarId,
) -> Vec<Stmt> {
    let mut out = vec![
        Stmt::StoreShared {
            array: sh,
            index: lane,
            value: src,
        },
        Stmt::Barrier,
    ];

    // One level of either sweep: the guarded index, computed by every lane,
    // and the body only the lanes that own a node execute. The barrier sits
    // outside that guard - it is the level's, not the node's.
    let level = |lo: &mut Lowerer, out: &mut Vec<Stmt>, step: u32| {
        let scaled = push_expr(lo, out, "sc", VarKind::Idx, LExpr::IMulC(lane, 2 * step));
        let idx = push_expr(
            lo,
            out,
            "ix",
            VarKind::Idx,
            LExpr::IAddC(scaled, 2 * step - 1),
        );
        let cond = push_expr(
            lo,
            out,
            "in",
            VarKind::Bool,
            LExpr::ICmpC {
                op: rir_core::CmpOp::Lt,
                var: idx,
                c: lanes,
            },
        );
        (idx, cond)
    };

    let mut step = 1u32;
    while step < lanes {
        let (idx, cond) = level(lo, &mut out, step);
        let mut body = Vec::new();
        let below = push_expr(lo, &mut body, "lo", VarKind::Idx, LExpr::ISubC(idx, step));
        let a = lo.new_var("a", VarKind::F32);
        body.push(Stmt::LoadShared {
            dst: a,
            array: sh,
            index: below,
            width: 1,
        });
        let b = lo.new_var("b", VarKind::F32);
        body.push(Stmt::LoadShared {
            dst: b,
            array: sh,
            index: idx,
            width: 1,
        });
        let c = push_expr(
            lo,
            &mut body,
            "c",
            VarKind::F32,
            LExpr::Combine { op, lhs: a, rhs: b },
        );
        body.push(Stmt::StoreShared {
            array: sh,
            index: idx,
            value: c,
        });
        out.push(Stmt::If { cond, body });
        out.push(Stmt::Barrier);
        step <<= 1;
    }

    // The identity at the root, which is what makes the prefix exclusive.
    let last = push_expr(
        lo,
        &mut out,
        "root",
        VarKind::Idx,
        LExpr::ConstIdx(lanes - 1),
    );
    let ident = lo.new_var("id", VarKind::F32);
    out.push(Stmt::LaneZero {
        lane,
        body: vec![
            Stmt::InitAcc { acc: ident, op },
            Stmt::StoreShared {
                array: sh,
                index: last,
                value: ident,
            },
        ],
    });
    out.push(Stmt::Barrier);

    let mut step = lanes / 2;
    while step > 0 {
        let (idx, cond) = level(lo, &mut out, step);
        let mut body = Vec::new();
        let below = push_expr(lo, &mut body, "lo", VarKind::Idx, LExpr::ISubC(idx, step));
        let swap = lo.new_var("swap", VarKind::F32);
        body.push(Stmt::LoadShared {
            dst: swap,
            array: sh,
            index: below,
            width: 1,
        });
        let here = lo.new_var("here", VarKind::F32);
        body.push(Stmt::LoadShared {
            dst: here,
            array: sh,
            index: idx,
            width: 1,
        });
        body.push(Stmt::StoreShared {
            array: sh,
            index: below,
            value: here,
        });
        let c = push_expr(
            lo,
            &mut body,
            "c",
            VarKind::F32,
            LExpr::Combine {
                op,
                lhs: here,
                rhs: swap,
            },
        );
        body.push(Stmt::StoreShared {
            array: sh,
            index: idx,
            value: c,
        });
        out.push(Stmt::If { cond, body });
        out.push(Stmt::Barrier);
        step >>= 1;
    }

    out.push(Stmt::LoadShared {
        dst,
        array: sh,
        index: lane,
        width: 1,
    });
    out
}
