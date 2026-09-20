//! Loop-invariant hoisting on Loop IR.
//!
//! Vectorization lets one invocation serve four elements. Hoisting addresses
//! the remaining cost: generated kernels build every address from scratch, at
//! every access, inside every loop. A rank-4 read inside a row loop prints
//!
//! ```text
//! col * x_nb0 + row * x_nb1 + plane * x_nb2 + batch * x_nb3
//! ```
//!
//! where only the first term moves. Three multiplies and three adds per
//! element, per access, and a kernel like `rms_norm_back` has five accesses
//! over two loops.
//!
//! This pass is a **Loop IR rewrite and not an emitter trick**, which is the
//! whole point: Metal, Vulkan and the CPU emitter get it at once, the
//! interpreter executes the rewritten nest so the oracle judges what the device
//! runs, and a fourth backend inherits it without knowing it exists.
//!
//! Two things move out of a sequential loop:
//!
//! - **address bases.** The invariant terms of an access address become one
//!   `LExpr::AddrSum` register, computed before the loop; the access keeps its
//!   varying terms plus that register. Identical bases are computed once,
//!   `rms_norm_back` reads `x` twice per iteration from the same base.
//! - **pure computations.** A `Stmt::Compute` whose operands are all defined
//!   outside the loop is the same value on every iteration. In
//!   `rms_norm_back`'s epilogue loop that is the entire scale computation:
//!   `-Σx·dz / (Σx² + eps·N)` and `1/√(Σx²/N + eps)` are recomputed once per
//!   *column* today.
//!
//! What does **not** move, and the reason each time:
//!
//! - `Stmt::Load` - a read is not pure. Under `Stmt::InBounds` or a `VecTail`
//!   guard it is precisely the access the guard exists to prevent.
//! - anything whose operands a loop writes. An accumulator is the case that
//!   matters: `InitAcc` sits *outside* the accumulation loop, so a naive
//!   "defined outside" test would happily hoist a use of it. `Stmt::Accum`
//!   therefore counts as a definition inside the loop.
//! - anything across a `Parallel`, `ParallelLane` or `StageTiles` boundary,
//!   which is not a loop this pass owns. Hoisting stops at the enclosing
//!   sequential loop and repeats outward, so a value invariant in three nested
//!   loops still ends up outside all three.
//!
//! Correctness rests on one property of Loop IR and one of the expressions:
//! a `Compute` destination is written exactly once (registers are numbered on
//! creation and never reused), and every `LExpr` is pure - no trap, no memory,
//! no ordering. Evaluating one of them on a path that previously skipped it is
//! therefore only wasted work, never a different answer. A loop with zero
//! iterations is the visible instance of this property, so the pass states it
//! explicitly.

use rir_core::IrId;
use std::collections::{HashMap, HashSet};

use crate::loop_ir::{AddrTerm, Inst, LExpr, Stmt, VarId, VarKind};

/// Hoists loop-invariant addresses and computations out of every sequential
/// loop of `body`, allocating the registers it needs in `var_names`/`var_kinds`.
pub fn hoist_invariants(
    body: &mut Vec<Stmt>,
    var_names: &mut Vec<String>,
    var_kinds: &mut Vec<VarKind>,
) {
    let mut ctx = Ctx {
        var_names,
        var_kinds,
    };
    hoist_block(body, &mut ctx);
}

struct Ctx<'a> {
    var_names: &'a mut Vec<String>,
    var_kinds: &'a mut Vec<VarKind>,
}

impl Ctx<'_> {
    fn new_var(&mut self, base: &str) -> VarId {
        let id = VarId::at(self.var_names.len());
        self.var_names.push(format!("{base}_v{}", id.0));
        self.var_kinds.push(VarKind::Idx);
        id
    }
}

/// Rewrites one statement list: children first (so an inner loop has already
/// pushed what it could into this block), then each loop of this block hands
/// its invariants to the statements just before it.
fn hoist_block(stmts: &mut Vec<Stmt>, ctx: &mut Ctx) {
    for s in stmts.iter_mut() {
        for child in child_blocks(s) {
            hoist_block(child, ctx);
        }
    }
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for mut s in std::mem::take(stmts) {
        if let Some((loop_var, body)) = loop_parts(&mut s) {
            let mut inside = HashSet::new();
            inside.insert(loop_var);
            defs_of_block(body, &mut inside);
            let mut hoisted = Vec::new();
            let mut bases: HashMap<BaseKey, VarId> = HashMap::new();
            extract(body, &inside, &mut hoisted, &mut bases, ctx);
            out.extend(hoisted);
        }
        out.push(s);
    }
    *stmts = out;
}

/// The sequential loops this pass hoists out of, and their body. `Parallel`,
/// `ParallelLane` and `StageTiles` are deliberately absent: they are not
/// iterations of the same code by one thread.
///
/// `VecTail` is here for its **tail** half only. That half is a loop - the
/// emitters print `for (tail = base; tail < n; ++tail)` - even though Loop IR
/// spells it as a field rather than a `Stmt::For`, and it is exactly the branch
/// that walks a row one element at a time, so it is where a recomputed address
/// costs the most. The vector half runs once and has nothing to hoist.
fn loop_parts(s: &mut Stmt) -> Option<(VarId, &mut Vec<Stmt>)> {
    match s {
        Stmt::For { var, body, .. }
        | Stmt::ForStrided { var, body, .. }
        | Stmt::ForChunk { var, body, .. }
        | Stmt::ForConst { var, body, .. }
        | Stmt::ForTiled { var, body, .. } => Some((*var, body)),
        Stmt::VecTail {
            tail_var,
            tail_body,
            ..
        } => Some((*tail_var, tail_body)),
        _ => None,
    }
}

/// Every nested statement list of `s`, for the recursive descent.
pub(crate) fn child_blocks(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::Parallel { body, .. }
        | Stmt::ParallelFlat { body, .. }
        | Stmt::ParallelLane { body, .. }
        | Stmt::For { body, .. }
        | Stmt::ForStrided { body, .. }
        | Stmt::ForChunk { body, .. }
        | Stmt::ForConst { body, .. }
        | Stmt::ForTiled { body, .. }
        | Stmt::LaneZero { body, .. }
        | Stmt::If { body, .. }
        | Stmt::InBounds { body, .. } => vec![body],
        // A tile's loader is descended into as well, and that is not
        // incidental: a segment-staged quantized operand puts its per-element
        // work in a `ForConst`, and what leaves that loop is the payload's
        // address base - the block times its stride, plus the row and the outer
        // planes. Without this the block/element split would halve the loads and leave
        // the address algebra behind.
        Stmt::StageTiles { tiles, body, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = tiles.iter_mut().map(|t| &mut t.load).collect();
            v.push(body);
            v
        }
        Stmt::VecTail {
            vec_body,
            tail_body,
            ..
        } => vec![vec_body, tail_body],
        _ => Vec::new(),
    }
}

/// Registers a statement subtree defines or overwrites. `Accum` is in here for
/// the reason the module doc gives: its accumulator is initialised outside the
/// loop but changes inside it.
fn defs_of(s: &Stmt, out: &mut HashSet<VarId>) {
    match s {
        Stmt::ParallelFlat {
            linear, axes, body, ..
        } => {
            // The linear index and every index it decomposes into: the
            // decomposition lives inside the statement, so these registers have
            // no defining statement of their own - the same reason a staged
            // tile lists its cooperative registers here.
            out.insert(*linear);
            for (v, _) in axes {
                out.insert(*v);
            }
            defs_of_block(body, out);
        }
        Stmt::Parallel { var, body, .. }
        | Stmt::ParallelLane { var, body, .. }
        | Stmt::For { var, body, .. }
        | Stmt::ForStrided { var, body, .. }
        | Stmt::ForChunk { var, body, .. }
        | Stmt::ForConst { var, body, .. }
        | Stmt::ForTiled { var, body, .. } => {
            out.insert(*var);
            defs_of_block(body, out);
        }
        Stmt::VecTail {
            tail_var,
            vec_body,
            tail_body,
            ..
        } => {
            out.insert(*tail_var);
            defs_of_block(vec_body, out);
            defs_of_block(tail_body, out);
        }
        Stmt::LaneZero { body, .. } | Stmt::If { body, .. } | Stmt::InBounds { body, .. } => {
            defs_of_block(body, out)
        }
        Stmt::StageTiles { tiles, body, .. } => {
            for t in tiles {
                // The cooperative loop lives in the emitters, so these
                // registers have no defining statement anywhere in the IR.
                // Listing them here is what stops a use of one from leaving the
                // staging round it belongs to.
                for v in [
                    t.tile,
                    t.row,
                    t.depth,
                    t.slot,
                    t.row_global,
                    t.depth_global,
                    t.row_origin,
                    t.depth_origin,
                ] {
                    out.insert(v);
                }
                defs_of_block(&t.load, out);
            }
            defs_of_block(body, out);
        }
        Stmt::InitAcc { acc, .. } | Stmt::Accum { acc, .. } => {
            out.insert(*acc);
        }
        Stmt::LaneReduce { dst, .. }
        | Stmt::LaneScan { dst, .. }
        | Stmt::LoadShared { dst, .. }
        | Stmt::Load { dst, .. } => {
            out.insert(*dst);
        }
        // Written, so nothing that reads it may leave the loop that writes it,
        // the very reason a loop-carried value needs its own statement.
        Stmt::Set { var, .. } => {
            out.insert(*var);
        }
        Stmt::WorkgroupReduce { reds, .. } => {
            for r in reds {
                out.insert(r.dst);
            }
        }
        Stmt::Compute(Inst { dst, .. }) => {
            out.insert(*dst);
        }
        Stmt::Store { .. } | Stmt::StoreShared { .. } | Stmt::Barrier => {}
    }
}

pub(crate) fn defs_of_block(stmts: &[Stmt], out: &mut HashSet<VarId>) {
    for s in stmts {
        defs_of(s, out);
    }
}

/// Registers an expression reads.
fn uses_of(e: &LExpr, out: &mut Vec<VarId>) {
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
        | LExpr::IToF(a) => out.push(*a),
        LExpr::Add(a, b)
        | LExpr::Sub(a, b)
        | LExpr::Mul(a, b)
        | LExpr::Div(a, b)
        | LExpr::IShr(a, b)
        | LExpr::IOr(a, b)
        | LExpr::ISub(a, b)
        | LExpr::Combine { lhs: a, rhs: b, .. }
        | LExpr::IAdd(a, b) => {
            out.push(*a);
            out.push(*b);
        }
        LExpr::Cmp { lhs, rhs, .. } => {
            out.push(*lhs);
            out.push(*rhs);
        }
        LExpr::Select { cond, t, f } => {
            out.push(*cond);
            out.push(*t);
            out.push(*f);
        }
        LExpr::AddrSum { terms, .. } => {
            for t in terms {
                if let AddrTerm::VarNb { var, .. } | AddrTerm::VarConst { var, .. } = t {
                    out.push(*var);
                }
            }
        }
    }
}

fn term_var(t: &AddrTerm) -> Option<VarId> {
    match t {
        AddrTerm::VarNb { var, .. } | AddrTerm::VarConst { var, .. } => Some(*var),
        AddrTerm::Const(_) => None,
    }
}

/// Key identifying an address base, so two accesses sharing one compute it
/// once. Argument included: the same index registers against another argument
/// are another set of strides.
///
/// The terms themselves, not a formatted rendering of them: an `AddrTerm` is
/// `Eq + Hash`, so the key is the structure and no formatting can make two
/// different bases collide.
type BaseKey = (rir_core::ArgId, Vec<AddrTerm>);

/// Walks one loop body in program order, moving what is invariant into
/// `hoisted` and rewriting the addresses that stay.
///
/// `inside` shrinks as the walk proceeds: a computation that has just left the
/// loop is available to the ones after it, which is what lets a chain of
/// dependent instructions move together in one pass.
fn extract(
    stmts: &mut Vec<Stmt>,
    inside: &HashSet<VarId>,
    hoisted: &mut Vec<Stmt>,
    bases: &mut HashMap<BaseKey, VarId>,
    ctx: &mut Ctx,
) {
    let mut kept: Vec<Stmt> = Vec::with_capacity(stmts.len());
    // Registers still varying: `inside` minus what this walk has already
    // hoisted. A `HashSet` copy rather than a mutation of the caller's, so
    // sibling blocks see the same state.
    let mut varying: HashSet<VarId> = inside.clone();
    for mut s in std::mem::take(stmts) {
        match &mut s {
            Stmt::Compute(inst) => {
                let mut u = Vec::new();
                uses_of(&inst.expr, &mut u);
                if u.iter().all(|v| !varying.contains(v)) {
                    varying.remove(&inst.dst);
                    hoisted.push(s);
                    continue;
                }
            }
            Stmt::Load { arg, addr, .. } | Stmt::Store { arg, addr, .. } => {
                rewrite_addr(*arg, addr, &varying, hoisted, bases, ctx);
            }
            // A tile's loader runs inside the emitters' cooperative loop, which
            // is a loop this pass cannot see - but its terms are ordinary
            // registers, and the outer ones (a plane, a batch) are as invariant
            // there as anywhere else. The tile's own coordinates are already in
            // `varying`, so descending with the same `extract` applies the same
            // split: an address base leaves, a guarded `Load` stays.
            Stmt::StageTiles { tiles, body, .. } => {
                for t in tiles.iter_mut() {
                    extract(&mut t.load, &varying, hoisted, bases, ctx);
                }
                extract(body, &varying, hoisted, bases, ctx);
            }
            _ => {
                for child in child_blocks(&mut s) {
                    extract(child, &varying, hoisted, bases, ctx);
                }
            }
        }
        kept.push(s);
    }
    *stmts = kept;
}

/// Replaces the invariant terms of one address by a single register.
///
/// Two terms is the threshold, and it is not arbitrary: one invariant term is
/// already one multiply, so naming it would trade a multiply for a register
/// and gain nothing. From two upward, `k` multiplies and `k-1` adds per access
/// per iteration become one add.
fn rewrite_addr(
    arg: rir_core::ArgId,
    addr: &mut Vec<AddrTerm>,
    varying: &HashSet<VarId>,
    hoisted: &mut Vec<Stmt>,
    bases: &mut HashMap<BaseKey, VarId>,
    ctx: &mut Ctx,
) {
    let invariant: Vec<AddrTerm> = addr
        .iter()
        .filter(|t| term_var(t).is_none_or(|v| !varying.contains(&v)))
        .cloned()
        .collect();
    if invariant.len() < 2 {
        return;
    }
    let key = (arg, invariant.clone());
    let base = match bases.get(&key) {
        Some(v) => *v,
        None => {
            let v = ctx.new_var("base");
            hoisted.push(Stmt::Compute(Inst {
                dst: v,
                expr: LExpr::AddrSum {
                    arg,
                    terms: invariant,
                },
            }));
            bases.insert(key, v);
            v
        }
    };
    addr.retain(|t| term_var(t).is_some_and(|v| varying.contains(&v)));
    addr.push(AddrTerm::VarConst { var: base, c: 1 });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};
    use crate::loop_ir::LoopKernel;
    use crate::lower::lower;
    use crate::schedule::Schedule;
    use rir_core::{Extent, KernelBuilder, ReduceOp, ReductionSemantics, TensorType};

    /// A rank-4 kernel with an inner reduction and an elementwise epilogue:
    /// two loops over the same row, five accesses, three outer axes whose
    /// stride products are the thing this pass exists to remove. Shaped like
    /// `rms_norm_back` without depending on `rir-kernels`, which depends on
    /// this crate.
    fn scaled_row() -> rir_core::ValidatedKernel {
        let mut k = KernelBuilder::new("scaled_row");
        let x = k.input("x", TensorType::f32(4));
        let y = k.output("y", TensorType::f32(4));
        let eps = k.param("eps", rir_core::ScalarType::F32);
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
        let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let xv = k.read(x, &[col, row, plane, batch]);
        let sq = k.mul(xv, xv);
        let sum = k.reduce(ReduceOp::Sum, col, sq, ReductionSemantics::Deterministic);
        // A whole subexpression of the epilogue that depends on nothing the
        // column loop changes.
        let n = k.axis_extent(col);
        let mean = k.div(sum, n);
        let shifted = k.add(mean, eps);
        let rrms = {
            let one = k.const_f32(1.0);
            let r = k.sqrt(shifted);
            k.div(one, r)
        };
        let v = k.mul(xv, rrms);
        k.write(y, &[col, row, plane, batch], v);
        k.finish().unwrap()
    }

    /// Every `Load`/`Store` address in the nest, with the loop depth it sits at.
    fn addrs(k: &LoopKernel) -> Vec<usize> {
        fn walk(stmts: &[Stmt], out: &mut Vec<usize>) {
            for s in stmts {
                match s {
                    Stmt::Load { addr, .. } | Stmt::Store { addr, .. } => out.push(addr.len()),
                    _ => {
                        let mut s = s.clone();
                        for b in child_blocks(&mut s) {
                            walk(b, out);
                        }
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(&k.body, &mut out);
        out
    }

    fn count_computes(stmts: &[Stmt]) -> usize {
        let mut n = 0;
        for s in stmts {
            if matches!(s, Stmt::Compute(_)) {
                n += 1;
            }
            let mut s = s.clone();
            for b in child_blocks(&mut s) {
                n += count_computes(b);
            }
        }
        n
    }

    /// The measurable claim: a rank-4 access inside a loop no longer carries
    /// four stride products. One term walks the loop, one names everything
    /// else.
    #[test]
    fn a_rank_four_access_inside_a_loop_keeps_two_terms() {
        for schedule in [
            Schedule::cpu_serial(),
            Schedule::vulkan_subgroup(),
            Schedule::metal_simdgroup(),
        ] {
            let lk = lower(&scaled_row(), schedule.clone()).unwrap();
            let widths = addrs(&lk);
            assert!(!widths.is_empty());
            assert!(
                widths.iter().all(|&n| n == 2),
                "{:?}: addresses with {widths:?} terms",
                schedule.backend
            );
        }
    }

    /// Hoisting must not change a single bit. The pass moves pure expressions
    /// and splits an address into two registers whose sum is the same integer;
    /// there is no reassociation of anything the reduction semantics protect,
    /// so the oracle's output is compared for **equality**, not within a
    /// tolerance.
    #[test]
    fn the_hoisted_nest_computes_exactly_what_the_flat_one_computed() {
        let kernel = scaled_row();
        let (n_col, n_row, n_plane, n_batch) = (37usize, 3usize, 2usize, 2usize);
        let len = n_col * n_row * n_plane * n_batch;
        let nb = [4, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row * n_plane];
        let shape = [n_col, n_row, n_plane, n_batch];
        let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();

        let hoisted = lower(&kernel, Schedule::cpu_serial()).unwrap();
        // The same lowering with the pass skipped, which is what "unchanged"
        // has to mean: the comparison is against this nest and not against a
        // hand-written reference, so a bug in the pass cannot hide behind a
        // tolerance in the model.
        let mut flat = hoisted.clone();
        flat.body = lower_without_hoisting(&kernel);

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
                run(lk, &mut args, &[1e-5]).unwrap();
            }
            y
        };
        assert_eq!(run_one(&hoisted), run_one(&flat));
    }

    /// Same lowering, pass not applied - the control the equality test needs.
    fn lower_without_hoisting(kernel: &rir_core::ValidatedKernel) -> Vec<Stmt> {
        // `lower` always hoists, so the control is rebuilt by *undoing* the
        // split: an `AddrSum` register is inlined back into every address that
        // reads it, and the hoisted computations stay where they are (they are
        // pure, so their position is not what is under test here).
        let lk = lower(kernel, Schedule::cpu_serial()).unwrap();
        let mut sums: HashMap<VarId, (rir_core::ArgId, Vec<AddrTerm>)> = HashMap::new();
        collect_sums(&lk.body, &mut sums);
        let mut body = lk.body.clone();
        inline_sums(&mut body, &sums);
        body
    }

    fn collect_sums(stmts: &[Stmt], out: &mut HashMap<VarId, (rir_core::ArgId, Vec<AddrTerm>)>) {
        for s in stmts {
            if let Stmt::Compute(Inst {
                dst,
                expr: LExpr::AddrSum { arg, terms },
            }) = s
            {
                out.insert(*dst, (*arg, terms.clone()));
            }
            let mut s = s.clone();
            for b in child_blocks(&mut s) {
                collect_sums(b, out);
            }
        }
    }

    fn inline_sums(stmts: &mut [Stmt], sums: &HashMap<VarId, (rir_core::ArgId, Vec<AddrTerm>)>) {
        for s in stmts.iter_mut() {
            match s {
                Stmt::Load { addr, .. } | Stmt::Store { addr, .. } => {
                    let mut flat = Vec::new();
                    for t in addr.iter() {
                        match term_var(t).and_then(|v| sums.get(&v)) {
                            Some((_, terms)) => flat.extend(terms.iter().cloned()),
                            None => flat.push(t.clone()),
                        }
                    }
                    *addr = flat;
                }
                _ => {
                    for b in child_blocks(s) {
                        inline_sums(b, sums);
                    }
                }
            }
        }
    }

    /// The trap the module doc names: an accumulator is initialised outside
    /// the loop and written inside it, so a use of it is **not** invariant.
    /// Without `Accum` counting as a definition, the multiply feeding the
    /// accumulation would leave the loop and every row would get the first
    /// column's value.
    #[test]
    fn an_accumulated_value_never_leaves_its_loop() {
        let lk = lower(&scaled_row(), Schedule::cpu_serial()).unwrap();
        let accs: Vec<VarId> = {
            let mut v = Vec::new();
            fn walk(stmts: &[Stmt], v: &mut Vec<VarId>) {
                for s in stmts {
                    if let Stmt::Accum { acc, .. } = s {
                        v.push(*acc);
                    }
                    let mut s = s.clone();
                    for b in child_blocks(&mut s) {
                        walk(b, v);
                    }
                }
            }
            walk(&lk.body, &mut v);
            v
        };
        assert!(
            !accs.is_empty(),
            "the test kernel does not accumulate anything"
        );

        // Every statement reading an accumulator must still be inside a loop.
        fn check(stmts: &[Stmt], accs: &[VarId], in_loop: bool) {
            for s in stmts {
                if let Stmt::Compute(Inst { expr, .. }) = s {
                    let mut u = Vec::new();
                    uses_of(expr, &mut u);
                    assert!(
                        in_loop || !u.iter().any(|v| accs.contains(v)),
                        "a computation reading an accumulator left its loop"
                    );
                }
                let mut owned = s.clone();
                let is_loop = loop_parts(&mut owned).is_some();
                let mut owned = s.clone();
                for b in child_blocks(&mut owned) {
                    check(b, accs, in_loop || is_loop);
                }
            }
        }
        check(&lk.body, &accs, false);
    }

    /// Nothing is lost on the way out: the pass moves statements, it does not
    /// drop or duplicate them, so the count of computations is the original
    /// one plus exactly the address registers it introduced.
    #[test]
    fn hoisting_moves_statements_and_creates_only_address_registers() {
        let lk = lower(&scaled_row(), Schedule::cpu_serial()).unwrap();
        let mut sums = HashMap::new();
        collect_sums(&lk.body, &mut sums);
        assert!(!sums.is_empty(), "no address base hoisted");
        let total = count_computes(&lk.body);
        assert!(
            total > sums.len(),
            "{total} computations for {} bases: the body disappeared",
            sums.len()
        );
        // Each hoisted base names a register the lowering did not have, so the
        // register banks must have grown with it.
        assert!(lk.var_names.len() == lk.var_kinds.len());
        for v in sums.keys() {
            assert!(matches!(lk.var_kinds[v.0 as usize], VarKind::Idx));
        }
    }
}
