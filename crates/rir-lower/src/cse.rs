//! Common subexpressions and dead registers on Loop IR.
//!
//! `crate::hoist` moves what is invariant out of a loop and already shares two
//! *identical addresses inside one loop*. What it does not do - and what the
//! audit measured in the generated shaders - is recognise the same expression
//! twice at the same level of the nest:
//!
//! - `rms_norm` hoists the base of its two passes over `x` out of two sibling
//!   loops, and computes the same three stride products twice;
//! - `add_repeat/vec4` hoists `row % n_row_b` out of the vector half and out of
//!   the scalar tail, and both land in the same enclosing block.
//!
//! Two passes here, in this order, and both on the **finished** nest - after
//! widening, after hoisting - because that is the only point at which what the
//! emitters will print is what is being read:
//!
//! - **CSE.** A `Stmt::Compute` whose expression is already available in an
//!   enclosing or current block becomes a rename: the statement is dropped and
//!   every later use of its destination reads the register that already holds
//!   the value.
//! - **Pruning.** A `Stmt::Compute` nobody reads is dropped. Not cosmetic
//!   either: the segment loader answers a quantized read from the block
//!   header, which leaves the element's own axis register with no consumer.
//!
//! Both rest on the same two properties `hoist` relies on, and they are worth
//! restating because they are what makes the rewrite safe rather than plausible:
//! every `LExpr` is **pure** (no memory, no trap, no ordering), and a `Compute`
//! destination is written **exactly once** - registers are numbered on creation
//! and never reused. So two computations of one key on registers nothing
//! reassigns are the same value, and a value nothing reads is a value nothing
//! misses.
//!
//! What limits the reuse is **dominance**, and it is not a nicety: the emitters
//! declare a register where the statement sits, inside whatever braces it sits
//! in, so a register defined in the vector half of a `VecTail` does not exist in
//! the scalar half. Availability therefore descends into child blocks and never
//! crosses between siblings. And it is *invalidated* by anything a nested block
//! reassigns - an accumulator under `Stmt::Accum`, the coordinates an emitter
//! drives - which is the same list `hoist` pins for the same reason.

use std::collections::{HashMap, HashSet};

use crate::hoist::{child_blocks, defs_of_block};
use crate::loop_ir::{AddrTerm, Inst, LExpr, Stmt, VarId};

/// Structural key of a pure expression: the form, then its operands.
///
/// A discriminant plus a flat word list rather than a derived `Hash` on `LExpr`,
/// for one reason: `ConstF32` holds an `f32`, which is neither `Eq` nor `Hash`.
/// Keying it on its **bits** is the conservative reading - `0.0` and `-0.0` stay
/// distinct, two NaNs of the same payload merge - and it keeps this key total
/// over the expression language instead of special-casing one arm.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Op(u8, Vec<u32>),
    /// An address sum, keyed on the argument and its terms - the same key
    /// `hoist` uses for a base, so the two passes agree on what "the same
    /// address" means.
    Addr(rir_core::ArgId, Vec<AddrTerm>),
}

fn cmp_tag(op: rir_core::CmpOp) -> u32 {
    match op {
        rir_core::CmpOp::Gt => 0,
        rir_core::CmpOp::Ge => 1,
        rir_core::CmpOp::Lt => 2,
        rir_core::CmpOp::Le => 3,
        rir_core::CmpOp::Eq => 4,
        rir_core::CmpOp::Ne => 5,
    }
}

fn reduce_tag(op: rir_core::ReduceOp) -> u32 {
    match op {
        rir_core::ReduceOp::Sum => 0,
        rir_core::ReduceOp::Max => 1,
    }
}

fn lut_tag(table: rir_core::LutId) -> u32 {
    // The table's own symbol is its identity in every emitter; hashing its
    // pointer-free name keeps this independent of the enum's declaration order.
    table
        .symbol()
        .bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(u32::from(b)))
}

fn key_of(e: &LExpr) -> Key {
    let op = Key::Op;
    match e {
        LExpr::ConstF32(c) => op(0, vec![c.to_bits()]),
        LExpr::Param(p) => op(1, vec![p.0]),
        LExpr::AxisExtent(a) => op(2, vec![a.0]),
        LExpr::Copy(a) => op(3, vec![a.0]),
        LExpr::Add(a, b) => op(4, vec![a.0, b.0]),
        LExpr::Sub(a, b) => op(5, vec![a.0, b.0]),
        LExpr::Mul(a, b) => op(6, vec![a.0, b.0]),
        LExpr::Div(a, b) => op(7, vec![a.0, b.0]),
        LExpr::Sqrt(a) => op(8, vec![a.0]),
        LExpr::Exp(a) => op(9, vec![a.0]),
        LExpr::Tanh(a) => op(10, vec![a.0]),
        LExpr::Cmp { op: c, lhs, rhs } => Key::Op(11, vec![cmp_tag(*c), lhs.0, rhs.0]),
        LExpr::Select { cond, t, f } => op(12, vec![cond.0, t.0, f.0]),
        LExpr::IDivC(a, c) => op(13, vec![a.0, *c]),
        LExpr::IModC(a, c) => op(14, vec![a.0, *c]),
        LExpr::IMulC(a, c) => op(15, vec![a.0, *c]),
        LExpr::IAdd(a, b) => op(16, vec![a.0, b.0]),
        LExpr::IAndC(a, c) => op(17, vec![a.0, *c]),
        LExpr::IShrC(a, c) => op(18, vec![a.0, *c]),
        LExpr::IShr(a, b) => op(19, vec![a.0, b.0]),
        LExpr::IOr(a, b) => op(20, vec![a.0, b.0]),
        LExpr::IModAxis { var, axis } => op(21, vec![var.0, axis.0]),
        LExpr::Lut { table, idx } => op(22, vec![lut_tag(*table), idx.0]),
        LExpr::IToF(a) => op(23, vec![a.0]),
        LExpr::ConstIdx(c) => op(24, vec![*c]),
        LExpr::IAddC(a, c) => op(25, vec![a.0, *c]),
        LExpr::ISubC(a, c) => op(26, vec![a.0, *c]),
        LExpr::ISub(a, b) => op(27, vec![a.0, b.0]),
        LExpr::ICmpC { op: c, var, c: n } => Key::Op(28, vec![cmp_tag(*c), var.0, *n]),
        LExpr::Combine { op: o, lhs, rhs } => Key::Op(29, vec![reduce_tag(*o), lhs.0, rhs.0]),
        LExpr::AxisExtentIdx(a) => op(30, vec![a.0]),
        LExpr::AddrSum { arg, terms } => Key::Addr(*arg, terms.clone()),
    }
}

/// Shares dominating subexpressions and drops the computations nobody reads.
pub fn canonicalize(body: &mut Vec<Stmt>) {
    let mut rename: HashMap<VarId, VarId> = HashMap::new();
    let pinned = pinned_regs(body);
    cse_block(body, &HashMap::new(), &mut rename, &pinned);
    prune(body);
}

/// Registers a `TileStage` names, and which therefore may not be renamed away by
/// CSE even when the computation feeding them is a duplicate.
///
/// The emitters *declare* these - a tiled scan's `value` is written by lowering
/// in the load phase and declared by the emitter in the store phase - so a
/// register renamed on the statement while its declaration stays behind would
/// print a shader referring to a name that does not exist there. Cheaper to pin
/// them than to reason, each time a statement is added, about which of its fields
/// an emitter spells.
fn pinned_regs(body: &[Stmt]) -> HashSet<VarId> {
    fn walk(stmts: &[Stmt], out: &mut HashSet<VarId>) {
        for s in stmts {
            if let Stmt::StageTiles { tiles, .. } = s {
                for t in tiles {
                    for v in [t.tile, t.row, t.depth, t.row_global, t.depth_global, t.slot] {
                        out.insert(v);
                    }
                }
            }
            let mut owned = s.clone();
            for child in child_blocks(&mut owned) {
                walk(child, out);
            }
        }
    }
    let mut out = HashSet::new();
    walk(body, &mut out);
    out
}

/// Rewrites one block in program order, with the expressions available on entry.
///
/// `avail` is cloned into every child block rather than shared, which is the
/// dominance rule in one line: what a block computes is visible to the blocks it
/// contains and to nothing else.
fn cse_block(
    stmts: &mut Vec<Stmt>,
    avail: &HashMap<Key, VarId>,
    rename: &mut HashMap<VarId, VarId>,
    pinned: &HashSet<VarId>,
) {
    let mut here = avail.clone();
    let mut kept: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for mut s in std::mem::take(stmts) {
        // Uses first: an operand may have been renamed by an earlier statement,
        // and the key of this expression has to be the renamed one.
        for_each_use(&mut s, &mut |v: &mut VarId| {
            while let Some(&r) = rename.get(v) {
                if r == *v {
                    break;
                }
                *v = r;
            }
        });
        if let Stmt::Compute(Inst { dst, expr }) = &s {
            let key = key_of(expr);
            match here.get(&key) {
                Some(&existing) if !pinned.contains(dst) => {
                    rename.insert(*dst, existing);
                    continue;
                }
                // A pinned destination keeps its own statement; it does not even
                // publish its key, so the *other* one is the shared register.
                Some(_) => {}
                None => {
                    here.insert(key, *dst);
                }
            }
        }
        // A nested block may reassign what an available expression reads - an
        // accumulator, a coordinate the emitters drive - so those entries do not
        // survive the descent.
        let invalid: HashSet<VarId> = {
            let mut d = HashSet::new();
            let mut owned = s.clone();
            for child in child_blocks(&mut owned) {
                defs_of_block(child, &mut d);
            }
            d
        };
        if !child_blocks(&mut s).is_empty() {
            let inner: HashMap<Key, VarId> = here
                .iter()
                .filter(|(k, _)| !key_reads(k, &invalid))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            for child in child_blocks(&mut s) {
                cse_block(child, &inner, rename, pinned);
            }
            // And the rest of *this* block loses them too. An accumulator is the
            // case that matters, and it is the same one `hoist` names: a
            // computation reading `acc` before its loop is not the same value as
            // the same computation after it.
            here.retain(|k, _| !key_reads(k, &invalid));
        }
        kept.push(s);
    }
    *stmts = kept;
}

/// Whether a key reads any of these registers.
fn key_reads(k: &Key, regs: &HashSet<VarId>) -> bool {
    match k {
        // Every word of an `Op` key that is a register is a register, and the
        // constants among them cannot collide with one: a false positive here
        // costs a shared expression, never a wrong value.
        Key::Op(_, words) => words.iter().any(|w| regs.contains(&VarId(*w))),
        Key::Addr(_, terms) => terms.iter().any(|t| match t {
            AddrTerm::VarNb { var, .. } | AddrTerm::VarConst { var, .. } => regs.contains(var),
            AddrTerm::Const(_) => false,
        }),
    }
}

/// Drops every `Stmt::Compute` whose destination nothing reads, to a fixpoint:
/// removing one can leave its own operands unread.
fn prune(body: &mut Vec<Stmt>) {
    loop {
        let mut used = HashSet::new();
        collect_uses(body, &mut used);
        if !drop_unused(body, &used) {
            return;
        }
    }
}

fn collect_uses(stmts: &mut [Stmt], out: &mut HashSet<VarId>) {
    for s in stmts.iter_mut() {
        for_each_use(s, &mut |v: &mut VarId| {
            out.insert(*v);
        });
        for child in child_blocks(s) {
            collect_uses(child, out);
        }
    }
}

fn drop_unused(stmts: &mut Vec<Stmt>, used: &HashSet<VarId>) -> bool {
    let mut changed = false;
    for s in stmts.iter_mut() {
        for child in child_blocks(s) {
            changed |= drop_unused(child, used);
        }
    }
    let before = stmts.len();
    stmts.retain(|s| match s {
        Stmt::Compute(Inst { dst, .. }) => used.contains(dst),
        _ => true,
    });
    changed || stmts.len() != before
}

/// Every register a statement **reads**, as a mutable reference, excluding its
/// nested blocks.
///
/// Exhaustive on purpose - no wildcard arm - because both passes above are only
/// as safe as this list is complete: a use it forgets is a register CSE renames
/// halfway or pruning deletes outright.
///
/// Two families are listed conservatively as reads even where they are also
/// written, and for the same reason each time. An accumulator is read by its own
/// `Accum`. And every register a `TileStage` or a `ScanTiles` names is one the
/// **emitters** print: a tile's origin is a computation this pass may share, and
/// a tiled scan's `value` is written by lowering and read by an emitter's store
/// phase, so nothing here may drop it.
fn for_each_use(s: &mut Stmt, f: &mut impl FnMut(&mut VarId)) {
    match s {
        Stmt::Parallel { .. }
        | Stmt::ParallelFlat { .. }
        | Stmt::ParallelLane { .. }
        | Stmt::For { .. } => {}
        Stmt::VecTail { base, .. } => f(base),
        Stmt::ForStrided { start, .. } => f(start),
        Stmt::ForChunk { lane, .. } => f(lane),
        Stmt::ForConst { .. } | Stmt::ForTiled { .. } => {}
        Stmt::InitAcc { acc, .. } => f(acc),
        Stmt::Accum { acc, value, .. } => {
            f(acc);
            f(value);
        }
        Stmt::LaneReduce { src, .. } | Stmt::LaneScan { src, .. } => f(src),
        Stmt::WorkgroupReduce { reds, .. } => {
            for r in reds.iter_mut() {
                f(&mut r.src);
            }
        }
        Stmt::LaneZero { lane, .. } => f(lane),
        Stmt::StageTiles { tiles, .. } => {
            for t in tiles.iter_mut() {
                for v in [
                    &mut t.tile,
                    &mut t.row,
                    &mut t.depth,
                    &mut t.row_global,
                    &mut t.depth_global,
                    &mut t.row_origin,
                    &mut t.depth_origin,
                    &mut t.slot,
                ] {
                    f(v);
                }
            }
        }
        Stmt::StoreShared {
            array,
            index,
            value,
        } => {
            f(array);
            f(index);
            f(value);
        }
        Stmt::LoadShared { array, index, .. } => {
            f(array);
            f(index);
        }
        Stmt::Barrier => {}
        Stmt::If { cond, .. } => f(cond),
        // Both, and deliberately: the target of an assignment is written *and*
        // read - it was declared elsewhere, and pruning it because this
        // statement is its only mention would delete the loop-carried value.
        Stmt::Set { var, value } => {
            f(var);
            f(value);
        }
        Stmt::InBounds { bounds, .. } => {
            for (var, _) in bounds.iter_mut() {
                f(var);
            }
        }
        Stmt::Load { addr, .. } => addr_uses(addr, f),
        Stmt::Store {
            addr, value, bound, ..
        } => {
            addr_uses(addr, f);
            f(value);
            if let Some((var, _)) = bound {
                f(var);
            }
        }
        Stmt::Compute(Inst { expr, .. }) => expr_uses(expr, f),
    }
}

fn addr_uses(addr: &mut [AddrTerm], f: &mut impl FnMut(&mut VarId)) {
    for t in addr.iter_mut() {
        match t {
            AddrTerm::VarNb { var, .. } | AddrTerm::VarConst { var, .. } => f(var),
            AddrTerm::Const(_) => {}
        }
    }
}

fn expr_uses(e: &mut LExpr, f: &mut impl FnMut(&mut VarId)) {
    match e {
        LExpr::ConstF32(_)
        | LExpr::Param(_)
        | LExpr::AxisExtent(_)
        | LExpr::AxisExtentIdx(_)
        | LExpr::ConstIdx(_) => {}
        LExpr::Copy(a)
        | LExpr::Sqrt(a)
        | LExpr::Exp(a)
        | LExpr::Tanh(a)
        | LExpr::IDivC(a, _)
        | LExpr::IModC(a, _)
        | LExpr::IMulC(a, _)
        | LExpr::IAndC(a, _)
        | LExpr::IShrC(a, _)
        | LExpr::IAddC(a, _)
        | LExpr::ISubC(a, _)
        | LExpr::ICmpC { var: a, .. }
        | LExpr::Lut { idx: a, .. }
        | LExpr::IModAxis { var: a, .. }
        | LExpr::IToF(a) => f(a),
        LExpr::Add(a, b)
        | LExpr::Sub(a, b)
        | LExpr::Mul(a, b)
        | LExpr::Div(a, b)
        | LExpr::IShr(a, b)
        | LExpr::IOr(a, b)
        | LExpr::ISub(a, b)
        | LExpr::Combine { lhs: a, rhs: b, .. }
        | LExpr::IAdd(a, b) => {
            f(a);
            f(b);
        }
        LExpr::Cmp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        LExpr::Select { cond, t, f: fv } => {
            f(cond);
            f(t);
            f(fv);
        }
        LExpr::AddrSum { terms, .. } => addr_uses(terms, f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_ir::LoopKernel;
    use crate::lower::lower;
    use crate::schedule::Schedule;
    use rir_core::{Extent, KernelBuilder, ReduceOp, ReductionSemantics, TensorType};

    /// Two passes over the same row: this is the shape of `rms_norm`, whose two
    /// loops hoist the **same** address base into the same block.
    fn two_pass_row() -> rir_core::ValidatedKernel {
        let mut k = KernelBuilder::new("two_pass");
        let x = k.input("x", TensorType::f32(4));
        let y = k.output("y", TensorType::f32(4));
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
        let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let xv = k.read(x, &[col, row, plane, batch]);
        let sq = k.mul(xv, xv);
        let sum = k.reduce(ReduceOp::Sum, col, sq, ReductionSemantics::Deterministic);
        let v = k.mul(xv, sum);
        k.write(y, &[col, row, plane, batch], v);
        k.finish().unwrap()
    }

    fn computes(lk: &LoopKernel) -> Vec<Key> {
        fn walk(stmts: &[Stmt], out: &mut Vec<Key>) {
            for s in stmts {
                if let Stmt::Compute(Inst { expr, .. }) = s {
                    out.push(key_of(expr));
                }
                let mut owned = s.clone();
                for b in child_blocks(&mut owned) {
                    walk(b, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(&lk.body, &mut out);
        out
    }

    /// The address base shared by both passes is computed only once. Without
    /// CSE, `hoist` extracts one per loop and both land in the same block.
    #[test]
    fn a_base_shared_by_two_sibling_loops_is_computed_once() {
        for schedule in [Schedule::cpu_serial(), Schedule::vulkan_subgroup()] {
            let lk = lower(&two_pass_row(), schedule.clone()).unwrap();
            let keys = computes(&lk);
            let mut seen = std::collections::HashSet::new();
            for k in &keys {
                assert!(
                    seen.insert(k.clone()),
                    "{:?}: an expression is computed twice in the same nest",
                    schedule.backend
                );
            }
            assert!(
                keys.iter().any(|k| matches!(k, Key::Addr(..))),
                "no base hoisted"
            );
        }
    }

    /// The canonicalized nest computes exactly what the other one computed: the
    /// pass renames and removes; it does not reassociate anything.
    #[test]
    fn canonicalizing_changes_no_value() {
        use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

        let kernel = two_pass_row();
        let lk = lower(&kernel, Schedule::cpu_serial()).unwrap();
        // The control: the same nest with the pass run a second time. It must be
        // a fixed point - if it changed anything else, the first application
        // would not have finished its work.
        let mut again = lk.clone();
        canonicalize(&mut again.body);

        let (n_col, n_row, n_plane, n_batch) = (13usize, 3usize, 2usize, 2usize);
        let len = n_col * n_row * n_plane * n_batch;
        let nb = [4, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row * n_plane];
        let shape = [n_col, n_row, n_plane, n_batch];
        let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.17).cos()).collect();
        let run_one = |lk: &LoopKernel| {
            let mut y = vec![0f32; len];
            {
                let mut args = [
                    BoundArg::In(TensorView {
                        data: &x,
                        shape,
                        nb,
                    }),
                    BoundArg::Out(TensorViewMut {
                        data: &mut y,
                        shape,
                        nb,
                    }),
                ];
                run(lk, &mut args, &[]).unwrap();
            }
            y
        };
        assert_eq!(run_one(&lk), run_one(&again));
        assert_eq!(computes(&lk).len(), computes(&again).len());
    }

    /// A register that nobody reads disappears, and none of what an emitter
    /// names disappears with it: the nest remains executable by the oracle.
    #[test]
    fn an_unread_computation_is_dropped() {
        let mut body = vec![Stmt::Compute(Inst {
            dst: VarId(0),
            expr: LExpr::ConstF32(1.0),
        })];
        canonicalize(&mut body);
        assert!(body.is_empty());

        // The tiled scan's element register is defined by the lowered IR and
        // consumed by a statement, so ordinary dead-code elimination must keep
        // it alive.
        let lk = lower(
            &{
                let mut k = KernelBuilder::new("cumsum");
                let x = k.input("x", TensorType::f32_2d());
                let y = k.output("y", TensorType::f32_2d());
                let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
                let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
                let xv = k.read(x, &[col, row]);
                let s = k.scan(
                    rir_core::ScanOp::Sum,
                    col,
                    rir_core::ScanDirection::Forward,
                    xv,
                );
                k.write(y, &[col, row], s);
                k.finish().unwrap()
            },
            Schedule::vulkan_tiled_scan(32, 4),
        )
        .unwrap();
        let Stmt::Parallel { body, .. } = &lk.body[0] else {
            panic!("root")
        };
        let Stmt::ParallelLane { body: lane, .. } = &body[0] else {
            panic!("ParallelLane")
        };
        assert!(
            lane.iter().any(|s| matches!(s, Stmt::ForTiled { .. })),
            "the tiled scan lowered to a tile loop"
        );
        // Every register a shared store writes is defined by a statement of the
        // nest - a computation, an accumulator, or a shared load. A pruned
        // value would leave one of them undefined, which is the failure this
        // test exists for.
        let mut defined: HashSet<VarId> = HashSet::new();
        let mut stored: Vec<VarId> = Vec::new();
        fn walk(stmts: &[Stmt], defined: &mut HashSet<VarId>, stored: &mut Vec<VarId>) {
            for s in stmts {
                match s {
                    Stmt::Compute(Inst { dst, .. })
                    | Stmt::LoadShared { dst, .. }
                    | Stmt::Load { dst, .. } => {
                        defined.insert(*dst);
                    }
                    Stmt::InitAcc { acc, .. } => {
                        defined.insert(*acc);
                    }
                    Stmt::StoreShared { value, .. } => stored.push(*value),
                    _ => {}
                }
                let mut owned = s.clone();
                for child in child_blocks(&mut owned) {
                    walk(child, defined, stored);
                }
            }
        }
        walk(&lk.body, &mut defined, &mut stored);
        assert!(!stored.is_empty(), "the scan writes no shared slot");
        for v in stored {
            assert!(
                defined.contains(&v),
                "{} is stored to shared memory and defined nowhere",
                lk.var_names[v.0 as usize]
            );
        }
    }
}
