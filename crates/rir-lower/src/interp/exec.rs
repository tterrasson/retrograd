//! Statement execution: the loop nest, the lane bank, and the loads.
//!
//! `exec_stmts` is the dispatcher and nothing else. It holds the **exhaustive
//! match** over `Stmt` - no wildcard, so a variant added to the Loop IR does not
//! compile until it is executed here - and hands each family to the function
//! below that knows it. The families are the ones the Loop IR already has:
//! the parallel nests, the sequential nests, the staged tile, the two accesses,
//! and the expression evaluator.
//!
//! **On the casts in this file** (docs/engineering/CONVERSIONS.md). There are about a
//! hundred and every one of them widens: `id.0 as usize` turns a `u32` register
//! identifier into an index, `*count as usize` a `u32` loop bound into one, and
//! the four `as f32` produce a value for the float bank. None can lose a bit on
//! a 64-bit target, so none is converted - the density of this file was a
//! measure of how many registers it indexes, not of any risk it carries.

use rir_core::{AxisId, CmpOp, ReduceOp};

use super::*;

// The eight arguments are the shape of `Stmt::Load` itself - kernel, args,
// registers, destination, binding, memory type, address, width - and grouping
// them in a struct would only move the same list: these are statement fields,
// not shared state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_load(
    k: &LoopKernel,
    args: &[BoundArg],
    regs: &mut Regs,
    dst: VarId,
    arg: rir_core::ArgId,
    ty: MemType,
    addr: &[AddrTerm],
    width: u32,
) -> Result<(), InterpError> {
    let arg_no = arg.0 as usize;
    let nb = args[arg_no].nb();
    // A widened access reads the `vlane`-th element of the vector, which is
    // `vlane` strides further along the contiguous dimension.
    let off = byte_addr(&nb, addr, regs) + if width > 1 { regs.vlane * nb[0] } else { 0 };
    let _ = k;
    let value = match (&args[arg_no], ty) {
        (BoundArg::In(v), MemType::F32) => {
            if !off.is_multiple_of(4) {
                return Err(InterpError::MisalignedAccess {
                    arg: arg_no,
                    byte_offset: off,
                });
            }
            let e = off / 4;
            if e >= v.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: v.data.len() * 4,
                });
            }
            v.data[e]
        }
        (BoundArg::InBytes(v), MemType::F16) => {
            // Match backends that index typed arrays: GLSL indexes
            // `float16_t[]`, not bytes, so a scale must have an even address.
            // Reject anything a shader could not read instead of accepting it
            // silently in the oracle.
            if !off.is_multiple_of(2) {
                return Err(InterpError::MisalignedAccess {
                    arg: arg_no,
                    byte_offset: off,
                });
            }
            if off + 2 > v.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: v.data.len(),
                });
            }
            f16_to_f32(u16::from_le_bytes([v.data[off], v.data[off + 1]]))
        }
        (BoundArg::InBytes(v), MemType::U8) => {
            if off >= v.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: v.data.len(),
                });
            }
            // Lands in the integer bank: the caller asked for raw bits.
            regs.i[dst.0 as usize] = v.data[off] as usize;
            return Ok(());
        }
        (BoundArg::InBytes(v), MemType::I8) => {
            if off >= v.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: v.data.len(),
                });
            }
            (v.data[off] as i8) as f32
        }
        _ => return Err(InterpError::BadAccessType { arg: arg_no }),
    };
    regs.f[dst.0 as usize] = value;
    Ok(())
}

// Mirror of `exec_load` above, and the same seven arguments for the same
// reason: they are the fields of `Stmt::Store`, so a struct would only move the
// list. It answers `Ok(true)` when the store was skipped by its per-component
// bound - the dispatcher's `continue`, returned instead of jumped to.
#[allow(clippy::too_many_arguments)]
fn exec_store(
    k: &LoopKernel,
    args: &mut [BoundArg],
    regs: &Regs,
    arg: rir_core::ArgId,
    ty: MemType,
    addr: &[AddrTerm],
    value: VarId,
    width: u32,
    bound: &Option<(VarId, AxisId)>,
) -> Result<(), InterpError> {
    // A per-component bound: this component of the vector is past the axis, so
    // it is not written at all.
    if let Some((var, axis)) = bound
        && regs.i[var.0 as usize] + regs.vlane >= axis_extent(k, *axis, args)
    {
        return Ok(());
    }
    let v = regs.f[value.0 as usize];
    let arg_no = arg.0 as usize;
    let nb = args[arg_no].nb();
    let off = byte_addr(&nb, addr, regs) + if width > 1 { regs.vlane * nb[0] } else { 0 };
    match (&mut args[arg_no], ty) {
        (BoundArg::Out(view), MemType::F32) => {
            if !off.is_multiple_of(4) {
                return Err(InterpError::MisalignedAccess {
                    arg: arg_no,
                    byte_offset: off,
                });
            }
            let e = off / 4;
            if e >= view.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: view.data.len() * 4,
                });
            }
            view.data[e] = v;
        }
        // The narrowing happens here and not in the caller, for the same reason
        // a shader narrows at its store: what the oracle must reproduce is the
        // *stored* value, rounding included.
        (BoundArg::OutBytes(view), MemType::F16) => {
            if !off.is_multiple_of(2) {
                return Err(InterpError::MisalignedAccess {
                    arg: arg_no,
                    byte_offset: off,
                });
            }
            if off + 2 > view.data.len() {
                return Err(InterpError::OutOfBounds {
                    arg: arg_no,
                    byte_offset: off,
                    len_bytes: view.data.len(),
                });
            }
            let b = f32_to_f16(v).to_le_bytes();
            view.data[off] = b[0];
            view.data[off + 1] = b[1];
        }
        _ => return Err(InterpError::BadAccessType { arg: arg_no }),
    }
    Ok(())
}

/// The expression evaluator: one `Inst`, on whichever register bank its kind
/// names.
///
/// Split from the dispatcher because it is the one arm that is not a statement
/// at all - it matches `LExpr`, a second and larger enumeration, and mixing the
/// two matches in one function was most of that function's length.
fn exec_compute(
    k: &LoopKernel,
    args: &[BoundArg],
    regs: &mut Regs,
    params: &[f32],
    dst: VarId,
    expr: &LExpr,
) -> Result<(), InterpError> {
    let d = dst.0 as usize;
    match expr {
        LExpr::IDivC(a, c) => regs.i[d] = regs.i[a.0 as usize] / *c as usize,
        LExpr::IModC(a, c) => regs.i[d] = regs.i[a.0 as usize] % *c as usize,
        LExpr::IMulC(a, c) => regs.i[d] = regs.i[a.0 as usize] * *c as usize,
        LExpr::IAdd(a, b) => regs.i[d] = regs.i[a.0 as usize] + regs.i[b.0 as usize],
        LExpr::IAndC(a, c) => regs.i[d] = regs.i[a.0 as usize] & *c as usize,
        LExpr::IShrC(a, c) => regs.i[d] = regs.i[a.0 as usize] >> *c as usize,
        LExpr::IShr(a, b) => regs.i[d] = regs.i[a.0 as usize] >> regs.i[b.0 as usize],
        LExpr::IOr(a, b) => regs.i[d] = regs.i[a.0 as usize] | regs.i[b.0 as usize],
        // The table is printed as floats by every emitter, so the oracle reads
        // it as floats too: an `i8` table converted at a different moment on
        // either side would be the one difference a bit-exact parity test could
        // not see through.
        LExpr::Lut { table, idx } => {
            let t = table.values();
            let i = regs.i[idx.0 as usize];
            regs.f[d] = t.get(i).copied().ok_or(InterpError::LutOutOfRange {
                table: *table,
                index: i,
            })? as f32;
        }
        // Same extent the loop bounds use, so a folded read and a plain one
        // cannot disagree on where a row ends.
        LExpr::IModAxis { var, axis } => {
            regs.i[d] = regs.i[var.0 as usize] % axis_extent(k, *axis, args).max(1)
        }
        LExpr::IToF(a) => regs.f[d] = regs.i[a.0 as usize] as f32,
        // A copy is the one expression that exists on **both** banks, so it is
        // the one that has to ask which register it is writing. The emitters
        // never had the question - they print `const {ty} x = y;` from the
        // register kind - and the oracle silently evaluated every copy as F32
        // until a lowered scan named an index that way.
        LExpr::Copy(a) if k.var_kinds[d] == VarKind::Idx => regs.i[d] = regs.i[a.0 as usize],
        LExpr::ConstIdx(c) => regs.i[d] = *c as usize,
        LExpr::IAddC(a, c) => regs.i[d] = regs.i[a.0 as usize] + *c as usize,
        // Saturating, and that is not a convenience: both shading languages
        // compute this on `uint`, where the same underflow wraps. Neither
        // result is ever read - the only subtraction the lowered trees perform
        // is guarded by the level's own test - so what matters is that the
        // oracle does not panic where a device would quietly produce a number
        // nobody looks at.
        LExpr::ISubC(a, c) => regs.i[d] = regs.i[a.0 as usize].saturating_sub(*c as usize),
        LExpr::ISub(a, b) => regs.i[d] = regs.i[a.0 as usize].saturating_sub(regs.i[b.0 as usize]),
        LExpr::AxisExtentIdx(axis) => regs.i[d] = axis_extent(k, *axis, args),
        LExpr::ICmpC { op, var, c } => {
            let (a, b) = (regs.i[var.0 as usize], *c as usize);
            let t = match op {
                CmpOp::Gt => a > b,
                CmpOp::Ge => a >= b,
                CmpOp::Lt => a < b,
                CmpOp::Le => a <= b,
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
            };
            regs.f[d] = if t { 1.0 } else { 0.0 };
        }
        // The hoisted half of an address (ADR-2 section 7). Evaluated by the same
        // `byte_addr` the accesses use, so a split address and a whole one
        // cannot disagree.
        LExpr::AddrSum { arg, terms } => {
            let nb = args[arg.0 as usize].nb();
            regs.i[d] = byte_addr(&nb, terms, regs);
        }
        // Same source as a loop bound, so the oracle and the shaders divide a
        // mean by the same number.
        LExpr::AxisExtent(axis) => regs.f[d] = axis_extent(k, *axis, args) as f32,
        _ => {
            let rf = |v: &VarId| regs.f[v.0 as usize];
            let val = match expr {
                LExpr::ConstF32(c) => *c,
                LExpr::Param(p) => params[p.0 as usize],
                LExpr::Copy(v) => rf(v),
                LExpr::Add(a, b) => rf(a) + rf(b),
                LExpr::Sub(a, b) => rf(a) - rf(b),
                LExpr::Mul(a, b) => rf(a) * rf(b),
                LExpr::Div(a, b) => rf(a) / rf(b),
                LExpr::Sqrt(a) => rf(a).sqrt(),
                LExpr::Exp(a) => rf(a).exp(),
                LExpr::Tanh(a) => rf(a).tanh(),
                LExpr::Cmp { op, lhs, rhs } => {
                    let (a, b) = (rf(lhs), rf(rhs));
                    let c = match op {
                        CmpOp::Gt => a > b,
                        CmpOp::Ge => a >= b,
                        CmpOp::Lt => a < b,
                        CmpOp::Le => a <= b,
                        CmpOp::Eq => a == b,
                        CmpOp::Ne => a != b,
                    };
                    if c { 1.0 } else { 0.0 }
                }
                LExpr::Combine { op, lhs, rhs } => {
                    let (a, b) = (rf(lhs), rf(rhs));
                    match op {
                        ReduceOp::Sum => a + b,
                        ReduceOp::Max => a.max(b),
                    }
                }
                LExpr::Select { cond, t, f } => {
                    if rf(cond) != 0.0 {
                        rf(t)
                    } else {
                        rf(f)
                    }
                }
                LExpr::IDivC(..)
                | LExpr::IModC(..)
                | LExpr::IMulC(..)
                | LExpr::IAndC(..)
                | LExpr::IShrC(..)
                | LExpr::IShr(..)
                | LExpr::IOr(..)
                | LExpr::Lut { .. }
                | LExpr::IModAxis { .. }
                | LExpr::IAdd(..)
                | LExpr::IToF(..)
                | LExpr::AddrSum { .. }
                | LExpr::AxisExtent(..)
                | LExpr::AxisExtentIdx(..)
                | LExpr::ConstIdx(..)
                | LExpr::IAddC(..)
                | LExpr::ISubC(..)
                | LExpr::ISub(..)
                | LExpr::ICmpC { .. } => {
                    unreachable!()
                }
            };
            regs.f[d] = val;
        }
    }
    Ok(())
}

/// The parallel nests: `Parallel`, `ParallelFlat` and `VecTail`.
///
/// They form a family because all three decide **which invocation runs**, and
/// all three have to place `vlane` before they descend. Anything else in the
/// Loop IR takes `vlane` as given.
///
/// # Panics
///
/// On any other statement. The exhaustive match in [`exec_stmts`] is what
/// selects this function, so an unhandled variant fails to compile there rather
/// than reaching this arm.
fn exec_parallel(
    k: &LoopKernel,
    s: &Stmt,
    regs: &mut Regs,
    sh: &mut Shared,
    args: &mut [BoundArg],
    params: &[f32],
) -> Result<(), InterpError> {
    match s {
        Stmt::Parallel {
            var,
            axis,
            vector,
            body,
            ..
        } => {
            let n = axis_extent(k, *axis, args);
            let w = (*vector).max(1) as usize;
            // One iteration per *invocation*, so a vectorized axis is
            // walked in strides of `vector` - the same indices the grid the
            // manifest publishes would produce. The invocation's body then
            // runs once per component, with `vlane` saying which: that is
            // what a vector register *means*, spelled without a second
            // register bank (`Regs`).
            for i in (0..n).step_by(w) {
                regs.i[var.0 as usize] = i;
                if w == 1 {
                    // A scalar axis leaves `vlane` alone: it may be nested
                    // *inside* a vectorized one, and resetting it here would
                    // collapse the whole vector onto its first component.
                    exec_stmts(k, body, regs, sh, args, params)?;
                    continue;
                }
                for c in 0..w {
                    regs.vlane = c;
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
                regs.vlane = 0;
            }
        }
        // The flattened dispatch. The oracle walks the
        // *same* linear space the device does - one iteration per
        // invocation, decomposed by plain division - so that the judgement
        // covers the decomposition itself and not only the body under it.
        // Plain `/` and `%` here against the emitters' magic numbers is
        // deliberate: the two must agree exactly, and
        // `the_magic_numbers_divide_exactly` is what proves they do.
        Stmt::ParallelFlat {
            linear,
            axes,
            decompose,
            vector,
            body,
            // The level says which hardware index the device reads its linear
            // one from; the oracle has no hardware and walks the same space
            // either way.
            level: _,
        } => {
            let w = (*vector).max(1);
            let divisors: Vec<usize> = axes
                .iter()
                .enumerate()
                .map(|(d, (_, ax))| {
                    let n = axis_extent(k, *ax, args);
                    if d == 0 { n.div_ceil(w as usize) } else { n }
                })
                .collect();
            let total: usize = divisors.iter().product();
            for t in 0..total {
                regs.i[linear.0 as usize] = t;
                // The oracle decomposes exactly when the shader does. Assigning
                // the index registers anyway under linear addressing would make
                // the oracle *define* what the device leaves undefined, and a
                // body that read one would agree here and read a garbage
                // register there.
                let mut r = t;
                for (d, (var, _)) in axes.iter().enumerate().filter(|_| *decompose) {
                    let i = if d + 1 == axes.len() {
                        r
                    } else {
                        let q = r % divisors[d];
                        r /= divisors[d];
                        q
                    };
                    regs.i[var.0 as usize] = if d == 0 { i * w as usize } else { i };
                }
                if w == 1 {
                    exec_stmts(k, body, regs, sh, args, params)?;
                    continue;
                }
                for c in 0..w as usize {
                    regs.vlane = c;
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
                regs.vlane = 0;
            }
        }
        Stmt::VecTail {
            base,
            axis,
            width,
            vec_body,
            tail_var,
            tail_body,
        } => {
            let n = axis_extent(k, *axis, args);
            let start = regs.i[base.0 as usize];
            if start + *width as usize <= n {
                // The enclosing `Parallel` already runs this once per
                // component, with `vlane` set.
                exec_stmts(k, vec_body, regs, sh, args, params)?;
            } else if regs.vlane == 0 {
                // The tail belongs to no component: run it on the first
                // pass only, or every element would be written `width`
                // times over.
                for t in start..n {
                    regs.i[tail_var.0 as usize] = t;
                    exec_stmts(k, tail_body, regs, sh, args, params)?;
                }
            }
        }
        other => unreachable!("exec_parallel: not a parallel nest - {other:?}"),
    }
    Ok(())
}

/// The sequential nests: `For`, `ForStrided`, `ForChunk`, `ForConst` and
/// `ForTiled`.
///
/// One family and not five arms: each sets one index register and re-enters the
/// dispatcher, and what separates them is only where the bounds come from - an
/// axis extent, a lane's share, a constant, a tile step.
///
/// # Panics
///
/// On any other statement, for the reason given on [`exec_parallel`].
fn exec_loop(
    k: &LoopKernel,
    s: &Stmt,
    regs: &mut Regs,
    sh: &mut Shared,
    args: &mut [BoundArg],
    params: &[f32],
) -> Result<(), InterpError> {
    match s {
        Stmt::For {
            var,
            axis,
            reverse,
            body,
        } => {
            let n = axis_extent(k, *axis, args);
            let exec_one = |i: usize,
                            regs: &mut Regs,
                            sh: &mut Shared,
                            args: &mut [BoundArg]|
             -> Result<(), InterpError> {
                regs.i[var.0 as usize] = i;
                exec_stmts(k, body, regs, sh, args, params)
            };
            if *reverse {
                for i in (0..n).rev() {
                    exec_one(i, regs, sh, args)?;
                }
            } else {
                for i in 0..n {
                    exec_one(i, regs, sh, args)?;
                }
            }
        }
        Stmt::ForStrided {
            var,
            axis,
            start,
            step,
            body,
        } => {
            let n = axis_extent(k, *axis, args);
            let mut i = regs.i[start.0 as usize];
            while i < n {
                regs.i[var.0 as usize] = i;
                exec_stmts(k, body, regs, sh, args, params)?;
                i += *step as usize;
            }
        }
        Stmt::ForChunk {
            var,
            axis,
            lane,
            lanes,
            reverse,
            body,
        } => {
            let n = axis_extent(k, *axis, args);
            let (start, end) = chunk_range(n, regs.i[lane.0 as usize], *lanes, *reverse);
            if *reverse {
                for i in (start..end).rev() {
                    regs.i[var.0 as usize] = i;
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
            } else {
                for i in start..end {
                    regs.i[var.0 as usize] = i;
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
            }
        }
        Stmt::ForConst { var, count, body } => {
            for i in 0..*count as usize {
                regs.i[var.0 as usize] = i;
                exec_stmts(k, body, regs, sh, args, params)?;
            }
        }
        Stmt::ForTiled {
            var,
            axis,
            step,
            body,
        } => {
            let n = axis_extent(k, *axis, args);
            let mut i = 0usize;
            while i < n {
                regs.i[var.0 as usize] = i;
                exec_stmts(k, body, regs, sh, args, params)?;
                i += (*step).max(1) as usize;
            }
        }
        other => unreachable!("exec_loop: not a sequential nest - {other:?}"),
    }
    Ok(())
}

/// `StageTiles`: fills every staged tile, then runs the body that reads them.
///
/// On its own because it is the only statement that *writes shared memory
/// before running anything*, and because the zero-fill and the in-bounds guard
/// under it are the contract the consumer relies on - a reader of the
/// dispatcher should find them named, not inlined between two loop arms.
fn exec_stage_tiles(
    k: &LoopKernel,
    tiles: &[TileStage],
    body: &[Stmt],
    regs: &mut Regs,
    sh: &mut Shared,
    args: &mut [BoundArg],
    params: &[f32],
) -> Result<(), InterpError> {
    for t in tiles {
        let n_rows = t.n_rows as usize;
        let row_n = axis_extent(k, t.row_axis, args);
        let depth_n = axis_extent(k, t.depth_axis, args);
        let span = t.span.max(1) as usize;
        for l in (0..t.len() as usize).step_by(span) {
            let (r, d) = (l % n_rows, l / n_rows);
            regs.i[t.row.0 as usize] = r;
            regs.i[t.depth.0 as usize] = d;
            regs.i[t.slot.0 as usize] = l;
            let m = regs.i[t.row_origin.0 as usize] + r;
            let kg = regs.i[t.depth_origin.0 as usize] + d;
            regs.i[t.row_global.0 as usize] = m;
            regs.i[t.depth_global.0 as usize] = kg;
            // Outside the tensor the tile holds zero, so the consumer needs no
            // predicate - and neither does the oracle, which is the point of
            // staging it here rather than of guarding every read below. The
            // loader runs only inside, exactly as the emitters print it: a
            // quantized decoder would otherwise address a block past the
            // tensor. It writes its own segment through `StoreShared`, so this
            // fills the zeros and steps aside.
            for c in 0..span {
                sh[t.tile.0 as usize][l + c] = 0.0;
            }
            if m < row_n && kg < depth_n {
                exec_stmts(k, &t.load, regs, sh, args, params)?;
            }
        }
    }
    exec_stmts(k, body, regs, sh, args, params)
}

/// The dispatcher, and deliberately nothing more.
///
/// The match below is **exhaustive and wildcard-free**: a variant added to
/// `Stmt` fails to compile here, which is the property the whole interpreter
/// leans on and the reason the families above are selected by pattern rather
/// than by a fall-through. The arms that stayed inline are the ones whose body
/// is shorter than the call that would replace it.
pub(crate) fn exec_stmts(
    k: &LoopKernel,
    stmts: &[Stmt],
    regs: &mut Regs,
    sh: &mut Shared,
    args: &mut [BoundArg],
    params: &[f32],
) -> Result<(), InterpError> {
    for s in stmts {
        match s {
            Stmt::Parallel { .. } | Stmt::ParallelFlat { .. } | Stmt::VecTail { .. } => {
                exec_parallel(k, s, regs, sh, args, params)?
            }
            Stmt::For { .. }
            | Stmt::ForStrided { .. }
            | Stmt::ForChunk { .. }
            | Stmt::ForConst { .. }
            | Stmt::ForTiled { .. } => exec_loop(k, s, regs, sh, args, params)?,
            Stmt::StageTiles { tiles, body, .. } => {
                exec_stage_tiles(k, tiles, body, regs, sh, args, params)?
            }
            Stmt::StoreShared {
                array,
                index,
                value,
            } => {
                let at = regs.i[index.0 as usize];
                sh[array.0 as usize][at] = regs.f[value.0 as usize];
            }
            Stmt::LoadShared {
                dst,
                array,
                index,
                width,
            } => {
                let at = regs.i[index.0 as usize] + if *width > 1 { regs.vlane } else { 0 };
                regs.f[dst.0 as usize] = sh[array.0 as usize][at];
            }
            // A barrier is free here, and the reason is the execution model
            // rather than a shortcut: inside a `ParallelLane` the oracle runs
            // one statement for every lane before it starts the next, so a
            // phase is finished for all lanes when the next one begins - which
            // is exactly what a barrier promises. Outside a lane body there is
            // one invocation and nothing to wait for.
            Stmt::Barrier => {}
            Stmt::If { cond, body } => {
                if regs.f[cond.0 as usize] != 0.0 {
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
            }
            Stmt::Set { var, value } => {
                let v = var.0 as usize;
                match k.var_kinds[v] {
                    VarKind::Idx => regs.i[v] = regs.i[value.0 as usize],
                    _ => regs.f[v] = regs.f[value.0 as usize],
                }
            }
            Stmt::InBounds { bounds, body } => {
                let inside = bounds
                    .iter()
                    .all(|(var, axis)| regs.i[var.0 as usize] < axis_extent(k, *axis, args));
                if inside {
                    exec_stmts(k, body, regs, sh, args, params)?;
                }
            }
            Stmt::ParallelLane { var, lanes, body } => {
                let mut lane_regs: Vec<Regs> = (0..*lanes)
                    .map(|l| {
                        let mut r = regs.clone();
                        r.i[var.0 as usize] = l as usize;
                        r
                    })
                    .collect();
                exec_lane_stmts(k, body, &mut lane_regs, sh, args, params)?;
            }
            Stmt::LaneReduce { .. }
            | Stmt::LaneScan { .. }
            | Stmt::WorkgroupReduce { .. }
            | Stmt::LaneZero { .. } => return Err(InterpError::MisplacedCollective),
            Stmt::InitAcc { acc, op } => {
                regs.f[acc.0 as usize] = match op {
                    ReduceOp::Sum => 0.0,
                    ReduceOp::Max => f32::NEG_INFINITY,
                };
            }
            Stmt::Accum { acc, op, value } => {
                let v = regs.f[value.0 as usize];
                let a = regs.f[acc.0 as usize];
                regs.f[acc.0 as usize] = match op {
                    ReduceOp::Sum => a + v,
                    ReduceOp::Max => a.max(v),
                };
            }
            Stmt::Load {
                dst,
                arg,
                ty,
                addr,
                width,
            } => {
                exec_load(k, args, regs, *dst, *arg, *ty, addr, *width)?;
            }
            Stmt::Store {
                arg,
                ty,
                addr,
                value,
                width,
                bound,
            } => {
                exec_store(k, args, regs, *arg, *ty, addr, *value, *width, bound)?;
            }
            Stmt::Compute(Inst { dst, expr }) => {
                exec_compute(k, args, regs, params, *dst, expr)?;
            }
        }
    }
    Ok(())
}

/// Combines one register across every lane in ascending lane order and
/// broadcasts the total, the semantics `LaneReduce` and one entry of a
/// `WorkgroupReduce` group share.
pub(crate) fn lane_total(lanes: &mut [Regs], op: ReduceOp, src: VarId, dst: VarId) {
    let total = match op {
        ReduceOp::Sum => lanes.iter().map(|r| r.f[src.0 as usize]).sum::<f32>(),
        ReduceOp::Max => lanes
            .iter()
            .map(|r| r.f[src.0 as usize])
            .fold(f32::NEG_INFINITY, f32::max),
    };
    for r in lanes.iter_mut() {
        r.f[dst.0 as usize] = total;
    }
}

/// `lane_total`, in two stages: one total per subgroup, then one over those
/// totals, broadcast to every lane.
pub(crate) fn lane_total_hierarchical(
    lanes: &mut [Regs],
    op: ReduceOp,
    src: VarId,
    dst: VarId,
    width: usize,
) {
    let identity = match op {
        ReduceOp::Sum => 0.0,
        ReduceOp::Max => f32::NEG_INFINITY,
    };
    let combine = |a: f32, b: f32| match op {
        ReduceOp::Sum => a + b,
        ReduceOp::Max => a.max(b),
    };
    let partials: Vec<f32> = lanes
        .chunks(width.max(1))
        .map(|sg| {
            sg.iter()
                .fold(identity, |a, r| combine(a, r.f[src.0 as usize]))
        })
        .collect();
    let total = partials.iter().fold(identity, |a, &p| combine(a, p));
    for r in lanes.iter_mut() {
        r.f[dst.0 as usize] = total;
    }
}

/// Executes a `ParallelLane` body. Non-collective instructions run lane by
/// lane with independent registers; collectives and `LaneZero` synchronize.
pub(crate) fn exec_lane_stmts(
    k: &LoopKernel,
    stmts: &[Stmt],
    lanes: &mut [Regs],
    sh: &mut Shared,
    args: &mut [BoundArg],
    params: &[f32],
) -> Result<(), InterpError> {
    for s in stmts {
        match s {
            // A group is executed entry by entry, and that *is* the semantics of
            // the grouped statement: sharing barriers changes when lanes meet,
            // not which values they combine, so the oracle of a group of two is
            // the oracle of two groups of one.
            Stmt::WorkgroupReduce {
                reds,
                subgroup: None,
                ..
            } => {
                for r in reds {
                    lane_total(lanes, r.op, r.src, r.dst);
                }
            }
            // The hierarchical tree combines the same values in a different
            // grouping, and the oracle reproduces the
            // grouping rather than the flat sum: subgroup by subgroup, then over
            // the subgroup totals. `Deterministic` allows the regrouping - it is
            // why the strategy is admissible at all - but an oracle that ignored
            // it would compare a two-stage sum with a one-stage one and charge
            // the difference to the shader.
            Stmt::WorkgroupReduce {
                reds,
                subgroup: Some(width),
                ..
            } => {
                for r in reds {
                    lane_total_hierarchical(lanes, r.op, r.src, r.dst, *width as usize);
                }
            }
            Stmt::LaneReduce { op, src, dst } => lane_total(lanes, *op, *src, *dst),
            Stmt::LaneScan { op, src, dst } => {
                // Exclusive prefix, lane by lane in increasing order - the same
                // order the hardware collective is specified to produce, and the
                // reason the blocked scan is reproducible.
                let mut acc = match op {
                    ReduceOp::Sum => 0.0,
                    ReduceOp::Max => f32::NEG_INFINITY,
                };
                for r in lanes.iter_mut() {
                    let v = r.f[src.0 as usize];
                    r.f[dst.0 as usize] = acc;
                    acc = match op {
                        ReduceOp::Sum => acc + v,
                        ReduceOp::Max => acc.max(v),
                    };
                }
            }
            Stmt::LaneZero { body, .. } => {
                exec_stmts(k, body, &mut lanes[0], sh, args, params)?;
            }
            // A barrier is what the execution model below already gives: every
            // statement of a lane body finishes for all lanes before the next
            // one starts. It is a no-op *because* of that, not instead of it.
            Stmt::Barrier => {}
            // Loops the workgroup walks **together**.
            //
            // A loop whose body holds a barrier cannot be run lane by lane:
            // lane 0 would finish every round before lane 1 started its first,
            // and the shared tile of round two would be read by a lane still in
            // round one. The tile loop of a lowered scan is exactly that loop.
            // So these three iterate here, at workgroup level, and their bodies
            // re-enter this function - which is what makes a barrier inside
            // them mean what it means on a device.
            //
            // `ForChunk` and `ForStrided` stay per-lane, in the arm below: their
            // bounds depend on the lane, so "the same iteration for every lane"
            // is not a thing they have, and no barrier may appear under one.
            Stmt::ForConst { var, count, body } => {
                for i in 0..*count as usize {
                    for r in lanes.iter_mut() {
                        r.i[var.0 as usize] = i;
                    }
                    exec_lane_stmts(k, body, lanes, sh, args, params)?;
                }
            }
            Stmt::ForTiled {
                var,
                axis,
                step,
                body,
            } => {
                let n = axis_extent(k, *axis, args);
                let mut i = 0usize;
                while i < n {
                    for r in lanes.iter_mut() {
                        r.i[var.0 as usize] = i;
                    }
                    exec_lane_stmts(k, body, lanes, sh, args, params)?;
                    i += (*step).max(1) as usize;
                }
            }
            Stmt::For {
                var,
                axis,
                reverse,
                body,
            } => {
                let n = axis_extent(k, *axis, args);
                let order: Vec<usize> = if *reverse {
                    (0..n).rev().collect()
                } else {
                    (0..n).collect()
                };
                for i in order {
                    for r in lanes.iter_mut() {
                        r.i[var.0 as usize] = i;
                    }
                    exec_lane_stmts(k, body, lanes, sh, args, params)?;
                }
            }
            other => {
                for r in lanes.iter_mut() {
                    exec_stmts(k, std::slice::from_ref(other), r, sh, args, params)?;
                }
            }
        }
    }
    Ok(())
}
