//! Widening a lowered nest to `vector_width` lanes per invocation, and the
//! shape analysis that decides which registers may be widened.

use crate::loop_ir::*;

use crate::lower::{Analysis, LowerError, Lowerer, lower_writes};

/// Vectorized elementwise lowering: one invocation
/// owns `width` consecutive elements of the contiguous axis.
///
/// ```text
/// parallel col = global_id.x · w        (grid, w indices per invocation)
///   if col + w <= n_col { vector body }  // w-wide loads, arithmetic, store
///   else { for t = col.. n_col { scalar body } }
/// ```
///
/// The two bodies are the **same writes lowered twice**, once with the axis
/// bound to the vector base and once to the tail variable, so neither is a
/// hand-written transcription of the other. The tail is a loop rather than a
/// masked vector because a partially out-of-bounds `float4` has no value a
/// contract could describe - and rejecting the shapes that produce one would
/// hand every row length that is not a multiple of four back to the native
/// kernel.
pub(crate) fn lower_vectorized(
    lo: &mut Lowerer,
    a: &Analysis,
    base: VarId,
    width: u32,
) -> Result<Vec<Stmt>, LowerError> {
    let axis = a.par_axes[0];

    // The vector half: the axis is bound to `base`, then every access whose
    // innermost index *is* `base` is widened.
    lo.axis_vars.insert(axis, base);
    let mut vec_body = lower_writes(lo, &a.row_writes)?;
    widen(lo, &mut vec_body, base, width)?;

    // The scalar half, on its own loop variable and its own registers.
    let tail_name = format!("{}_tail", lo.k.axes()[axis.0 as usize].name);
    let tail_var = lo.new_var(&tail_name, VarKind::Idx);
    lo.axis_vars.insert(axis, tail_var);
    let tail_body = lower_writes(lo, &a.row_writes)?;
    lo.axis_vars.insert(axis, base);

    Ok(vec![Stmt::VecTail {
        base,
        axis,
        width,
        vec_body,
        tail_var,
        tail_body,
    }])
}

/// Whether an instruction's result is a vector, given which of its operands
/// are - the componentwise rule, shared by every widening pass because it is a
/// property of the arithmetic and not of what seeded it.
///
/// A scalar operand broadcasts. The forms neither shading language spells
/// componentwise, or that would need a vector predicate, are errors: half a
/// widened body is worse than none.
/// What the widening pass made of one value: a scalar, a vector of floats, or
/// a vector of predicates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VecShape {
    F32,
    Bool,
}

pub(crate) fn expr_is_vec(
    expr: &LExpr,
    is_vec: &[Option<VecShape>],
) -> Result<Option<VecShape>, LowerError> {
    let vec_of = |v: &VarId| is_vec[v.0 as usize].is_some();
    let float_vec = |v: &VarId| is_vec[v.0 as usize] == Some(VecShape::F32);
    let f = |b: bool| if b { Some(VecShape::F32) } else { None };
    Ok(match expr {
        LExpr::Copy(x) | LExpr::Sqrt(x) | LExpr::Exp(x) | LExpr::Tanh(x) => f(vec_of(x)),
        LExpr::Add(x, y) | LExpr::Sub(x, y) | LExpr::Mul(x, y) | LExpr::Div(x, y) => {
            f(vec_of(x) || vec_of(y))
        }
        // A **vector** predicate is allowed: `select`/`mix` take one in
        // both languages. What is still refused is a mixed form nobody can
        // spell - a scalar branch under a vector predicate is fine (it
        // broadcasts), a *float* register used as a predicate is not.
        LExpr::Select { cond, t, f: fv } => {
            if float_vec(cond) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "floating-point vector predicate in a Select",
                });
            }
            f(vec_of(cond) || vec_of(t) || vec_of(fv))
        }
        LExpr::Cmp { lhs, rhs, .. } => {
            if vec_of(lhs) || vec_of(rhs) {
                Some(VecShape::Bool)
            } else {
                None
            }
        }
        LExpr::IDivC(x, _)
        | LExpr::IModC(x, _)
        | LExpr::IMulC(x, _)
        | LExpr::IAndC(x, _)
        | LExpr::IShrC(x, _)
        | LExpr::IToF(x)
        // A LUT read is indexed by an `Idx` register and produces one F32; a
        // vector index would be a gather, which neither emitter spells.
        | LExpr::Lut { idx: x, .. } => {
            if vec_of(x) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "integer arithmetic on a vector register",
                });
            }
            None
        }
        // A fold is index arithmetic like the rest: it takes an `Idx` register
        // and it is not widened. What it *does* forbid is a vectorized read of
        // the folded operand, and that is refused at the widening pass below,
        // four consecutive indices are four consecutive addresses only when the
        // fold is the identity, which is a runtime fact.
        LExpr::IModAxis { var: x, .. } => {
            if vec_of(x) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "integer arithmetic on a vector register",
                });
            }
            None
        }
        LExpr::IAddC(x, _) | LExpr::ISubC(x, _) | LExpr::ICmpC { var: x, .. } => {
            if vec_of(x) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "integer arithmetic on a vector register",
                });
            }
            None
        }
        // The scan trees are the only producer of these, and they are never
        // vectorized: a collective and a vector width are alternatives, not
        // layers. Widening one would need a componentwise shared access, which
        // the IR does not have.
        LExpr::Combine { lhs, rhs, .. } => {
            if vec_of(lhs) || vec_of(rhs) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "vector register under a collective combiner",
                });
            }
            None
        }
        LExpr::IAdd(x, y) | LExpr::IShr(x, y) | LExpr::IOr(x, y) | LExpr::ISub(x, y) => {
            if vec_of(x) || vec_of(y) {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "integer arithmetic on a vector register",
                });
            }
            None
        }
        // `AddrSum` is an `Idx` register and is produced after this pass by
        // `crate::hoist`, so it cannot be reached here; the arm is what keeps
        // the match total rather than a claim about widening addresses.
        LExpr::ConstF32(_)
        | LExpr::Param(_)
        | LExpr::AxisExtent(_)
        | LExpr::AxisExtentIdx(_)
        | LExpr::ConstIdx(_)
        | LExpr::AddrSum { .. } => None,
    })
}

/// Turns the scalar statements of one elementwise body into `width`-wide ones.
///
/// The rule is local and checkable: an access is widened exactly when its
/// innermost address term is `base · nb[0]` - the vectorized axis, at the
/// dimension whose stride the contract pins to one element. Everything else
/// stays scalar and broadcasts, which is what makes a per-row or per-plane read
/// (a bias, a scale) work without a special case.
///
/// Anything this cannot prove is an **error**, never a scalar fallback: a body
/// half-widened would compute three of four elements from the same address.
pub(crate) fn widen(
    lo: &mut Lowerer,
    stmts: &mut [Stmt],
    base: VarId,
    width: u32,
) -> Result<(), LowerError> {
    // Registers holding the vectorized axis **folded** into some extent.
    // An address whose innermost term is one of these
    // is widened exactly like one carrying `base` itself, under a claim the
    // registry publishes and a dispatch site evaluates: the fold must be the
    // identity, i.e. the repeated operand must have the same extent on the
    // contiguous dimension. `w` consecutive indices are then `w` consecutive
    // addresses, which is the same condition `nb[0] == elem_bytes` states for
    // an unfolded read - one more layout claim, not a new kind of claim.
    //
    // Refusing instead would be the visibly wrong trade: it is the *other*
    // dimensions a broadcast usually repeats, and a scalar kernel there costs
    // the vec4 deficit measured on exactly these shapes.
    let mut folded_base: Vec<bool> = vec![false; lo.var_names.len()];
    for s in stmts.iter() {
        if let Stmt::Compute(Inst {
            dst,
            expr: LExpr::IModAxis { var, .. },
        }) = s
            && *var == base
        {
            folded_base[dst.0 as usize] = true;
        }
    }

    // Whether the innermost index of this address is the vectorized axis (or
    // that axis folded), and whether `base` leaks into any other term - which
    // would make the address of element `base + c` something other than
    // `addr + c · nb[0]`.
    let indexed_by_base = |addr: &[AddrTerm]| -> Result<bool, LowerError> {
        // Two spellings of the same fact, and the second is linear addressing
        // there the whole address is `base · width ·
        // elem_bytes`, one term instead of a stride sum, and `base` is the
        // linear index rather than an axis's. The rule the widening needs is
        // unchanged - element `base + c` sits `c` elements further along - so
        // what differs is only how the address says so.
        let first = matches!(addr.first(),
            Some(AddrTerm::VarNb { var, dim: 0 })
                if *var == base || folded_base[var.0 as usize])
            || matches!(addr, [AddrTerm::VarConst { var, .. }] if *var == base);
        let elsewhere = addr.iter().skip(1).any(|t| match t {
            AddrTerm::VarNb { var, .. } | AddrTerm::VarConst { var, .. } => {
                *var == base || folded_base[var.0 as usize]
            }
            AddrTerm::Const(_) => false,
        });
        if elsewhere {
            return Err(LowerError::VectorWidthUnsupported {
                why: "the vectorized axis appears outside its address term",
            });
        }
        Ok(first)
    };

    let mut is_vec: Vec<Option<VecShape>> = vec![None; lo.var_names.len()];
    for s in stmts.iter_mut() {
        match s {
            Stmt::Load {
                dst,
                ty,
                addr,
                width: w,
                ..
            } => {
                if !indexed_by_base(addr)? {
                    continue;
                }
                // F32 and F16, and nothing else. Both are dense element types
                // whose `nb[0]` the contract pins to the element size, so
                // `width` consecutive indices really are consecutive addresses;
                // a byte view of a quantized block is neither.
                if !matches!(ty, MemType::F32 | MemType::F16) {
                    return Err(LowerError::VectorWidthUnsupported {
                        why: "non-dense read (quantized block) on the vectorized axis",
                    });
                }
                *w = width;
                is_vec[dst.0 as usize] = Some(VecShape::F32);
            }
            Stmt::Store {
                addr,
                value,
                width: w,
                ..
            } => {
                if indexed_by_base(addr)? {
                    *w = width;
                } else if is_vec[value.0 as usize].is_some() {
                    return Err(LowerError::VectorWidthUnsupported {
                        why: "scalar write of a vector value",
                    });
                }
            }
            Stmt::Compute(Inst { dst, expr }) => {
                if let Some(shape) = expr_is_vec(expr, &is_vec)? {
                    is_vec[dst.0 as usize] = Some(shape);
                }
            }
            _ => {
                return Err(LowerError::VectorWidthUnsupported {
                    why: "non-elementwise statement in the vectorized body",
                });
            }
        }
    }

    // Register kinds last, so a `Load` that turned out to be widened is typed
    // even though its statement carries no kind of its own.
    for (v, vectorized) in is_vec.iter().enumerate() {
        match vectorized {
            Some(VecShape::F32) => lo.var_kinds[v] = VarKind::Vec(width),
            Some(VecShape::Bool) => lo.var_kinds[v] = VarKind::VecBool(width),
            None => {}
        }
    }
    Ok(())
}

/// Propagates vector register widths through a tiled body.
///
/// Same componentwise rule as the elementwise `widen` (`expr_is_vec`), seeded
/// differently: there, by an address whose innermost term is the vectorized
/// axis; here, by the tile reads lowering already emitted wide. The
/// accumulator becomes a vector exactly when what it accumulates is one.
pub(crate) fn widen_tiled(
    lo: &mut Lowerer,
    stmts: &mut [Stmt],
    is_vec: &mut [Option<VecShape>],
    width: u32,
) -> Result<(), LowerError> {
    for s in stmts.iter_mut() {
        match s {
            Stmt::LoadShared { dst, width: w, .. } if *w > 1 => {
                is_vec[dst.0 as usize] = Some(VecShape::F32);
                lo.var_kinds[dst.0 as usize] = VarKind::Vec(width);
            }
            Stmt::Compute(Inst { dst, expr }) => {
                if let Some(shape) = expr_is_vec(expr, is_vec)? {
                    is_vec[dst.0 as usize] = Some(shape);
                    lo.var_kinds[dst.0 as usize] = match shape {
                        VecShape::F32 => VarKind::Vec(width),
                        VecShape::Bool => VarKind::VecBool(width),
                    };
                }
            }
            Stmt::Accum { acc, value, .. } if is_vec[value.0 as usize].is_some() => {
                is_vec[acc.0 as usize] = Some(VecShape::F32);
                lo.var_kinds[acc.0 as usize] = VarKind::Vec(width);
            }
            _ => {}
        }
    }
    Ok(())
}
