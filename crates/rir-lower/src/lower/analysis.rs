//! The shape analysis every strategy reads: which axes are parallel, which
//! is the inner one, and which writes belong to which phase.

use rir_core::IrId;
use rir_core::{ArgId, AxisId, Kernel, Op, ScanDirection, ValueId, depends_on_axis};

use crate::lower::LowerError;

pub(crate) struct Analysis {
    pub(crate) reduces: Vec<(ValueId, rir_core::ReduceOp, ValueId)>,
    pub(crate) scans: Vec<(ValueId, rir_core::ScanOp, ScanDirection, ValueId)>,
    pub(crate) par_axes: Vec<AxisId>,
    pub(crate) inner_axis: Option<AxisId>,
    /// Writes independent of the inner axis, emitted once per row.
    pub(crate) row_writes: Vec<(ArgId, Vec<ValueId>, ValueId)>,
    /// Writes dependent on the inner axis, emitted inside the loop.
    pub(crate) inner_writes: Vec<(ArgId, Vec<ValueId>, ValueId)>,
}

pub(crate) fn analyze(kernel: &Kernel) -> Result<Analysis, LowerError> {
    let mut reduces = Vec::new();
    let mut scans = Vec::new();
    let mut inner_axis: Option<AxisId> = None;
    let set_inner = |a: AxisId, inner: &mut Option<AxisId>| -> Result<(), LowerError> {
        match inner {
            None => {
                *inner = Some(a);
                Ok(())
            }
            Some(r) if *r != a => Err(LowerError::MultipleInnerAxes),
            _ => Ok(()),
        }
    };
    for (i, op) in kernel.ops().iter().enumerate() {
        match op {
            Op::Reduce {
                op, axis, value, ..
            } => {
                set_inner(*axis, &mut inner_axis)?;
                reduces.push((ValueId::at(i), *op, *value));
            }
            Op::Scan {
                op,
                axis,
                dir,
                value,
            } => {
                set_inner(*axis, &mut inner_axis)?;
                scans.push((ValueId::at(i), *op, *dir, *value));
            }
            _ => {}
        }
    }
    if !reduces.is_empty() && !scans.is_empty() {
        return Err(LowerError::MixedScanReduce);
    }
    if scans.iter().any(|(_, _, d, _)| *d != scans[0].2) {
        return Err(LowerError::MixedScanDirections);
    }

    // Only the axes the graph actually **walks** become loops. An indexed axis is
    // not necessarily walked: a `RepeatIndex`'s `over`
    // axis is declared for its *extent* alone - it is the divisor of a fold,
    // and opening a loop over it would traverse the repeated operand a second
    // time. It still gets its push constant and its
    // `arg_axes` row, which is exactly what a dispatcher needs to fill it.
    let walked: std::collections::HashSet<AxisId> = kernel
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::Index(a) => Some(*a),
            _ => None,
        })
        .collect();
    let par_axes: Vec<AxisId> = (0..kernel.axes().len())
        .map(AxisId::at)
        .filter(|a| Some(*a) != inner_axis && walked.contains(a))
        .collect();
    if kernel.axes().len() < 2 || par_axes.is_empty() {
        return Err(LowerError::UnsupportedAxisCount {
            got: kernel.axes().len(),
        });
    }

    let mut row_writes = Vec::new();
    let mut inner_writes = Vec::new();
    for op in kernel.ops() {
        if let Op::Write { tensor, idx, value } = op {
            let inner_dep = inner_axis.is_some_and(|ia| {
                idx.iter().any(|&i| depends_on_axis(kernel, i, ia))
                    || depends_on_axis(kernel, *value, ia)
            });
            let w = (*tensor, idx.clone(), *value);
            if inner_dep {
                inner_writes.push(w);
            } else {
                row_writes.push(w);
            }
        }
    }

    Ok(Analysis {
        reduces,
        scans,
        par_axes,
        inner_axis,
        row_writes,
        inner_writes,
    })
}
