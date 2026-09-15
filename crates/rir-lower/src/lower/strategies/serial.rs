//! The serial strategy: no collective, one invocation per parallel index.

use rir_core::ScanDirection;

use crate::loop_ir::*;

use crate::lower::{
    Analysis, LowerError, Lowerer, Traversal, lower_accum_loop, lower_writes, plan_segments,
    reduce_levels,
};

pub(crate) fn lower_serial(lo: &mut Lowerer, a: &Analysis) -> Result<Vec<Stmt>, LowerError> {
    let mut out = Vec::new();

    // Purely elementwise kernel: write at the parallel-nest level.
    let Some(inner_axis) = a.inner_axis else {
        let stmts = lower_writes(lo, &a.row_writes)?;
        out.extend(stmts);
        return Ok(out);
    };
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();

    if !a.scans.is_empty() {
        // Sequential scans keep state updates, inputs, and writes in one loop
        // traversed in the scan direction.
        let reverse = a.scans[0].2 == ScanDirection::Backward;
        let mut states = Vec::new();
        for (sv, sop, _, sin) in &a.scans {
            let acc = lo.new_var(&format!("scan{}", sv.0), VarKind::F32);
            let rop = match sop {
                rir_core::ScanOp::Sum => rir_core::ReduceOp::Sum,
            };
            out.push(Stmt::InitAcc { acc, op: rop });
            states.push((acc, rop, *sin, *sv));
        }

        let col_var = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col_var);
        lo.begin_scope(true);
        for (acc, rop, sin, sv) in &states {
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
        let loop_body = lo.take_body();
        out.push(Stmt::For {
            var: col_var,
            axis: inner_axis,
            reverse,
            body: loop_body,
        });

        if !a.row_writes.is_empty() {
            lo.axis_vars.remove(&inner_axis);
            let stmts = lower_writes(lo, &a.row_writes)?;
            out.extend(stmts);
        }
        return Ok(out);
    }

    if !a.reduces.is_empty() {
        let plan = plan_segments(lo.k, inner_axis);
        // Group reductions by dependency **level**. A reduction whose input
        // depends on another reduction runs in a later sequential pass, one
        // loop per level. Circular or same-level dependencies remain
        // `NestedReduce` errors.
        for group in reduce_levels(lo.k, &a.reduces).values() {
            let mut accs = Vec::new();
            for (rv, rop, rin) in group {
                let acc = lo.new_var(&format!("acc{}", rv.0), VarKind::F32);
                out.push(Stmt::InitAcc { acc, op: *rop });
                accs.push((acc, *rop, *rin, *rv));
            }

            // Results from earlier levels are visible. Referring to a level
            // not yet computed yields `NestedReduce` through an environment
            // lookup miss.
            out.extend(lower_accum_loop(
                lo,
                &plan,
                inner_axis,
                &accs,
                Traversal::Whole,
            )?);

            for (acc, _, _, rv) in &accs {
                lo.results_env.insert(*rv, *acc);
            }
        }
    }

    if !a.row_writes.is_empty() {
        lo.axis_vars.remove(&inner_axis);
        let stmts = lower_writes(lo, &a.row_writes)?;
        out.extend(stmts);
    }

    if !a.inner_writes.is_empty() {
        let col_var = lo.new_var(&inner_name, VarKind::Idx);
        lo.axis_vars.insert(inner_axis, col_var);
        let phase2 = lower_writes(lo, &a.inner_writes)?;
        out.push(Stmt::For {
            var: col_var,
            axis: inner_axis,
            reverse: false,
            body: phase2,
        });
    }

    Ok(out)
}
