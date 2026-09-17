//! IR-level autodiff (ADR-1 section 8): derive the backward pass by transposing the
//! SSA graph.
//!
//! The derived `{name}_grad` kernel takes the forward inputs plus one
//! `d<output>` gradient per output and produces one `d<input>` gradient per
//! input. Transposition rules:
//!
//! | Forward               | Backward                                      |
//! |-----------------------|-----------------------------------------------|
//! | `Map` (arithmetic)    | local derivative and chain rule               |
//! | uniform broadcast     | `Reduce(Sum)` over the inner axis             |
//! | `Reduce(Sum)`         | broadcast                                     |
//! | `Scan(Sum, dir)`      | `Scan(Sum, opposite_dir)`                     |
//! | `Reduce(Max)`         | **error** (manual argmax mask in v1)          |
//! | `Dequant`             | **error** (quantized gradients unsupported)   |
//!
//! A derived backward pass is never used before validation against a
//! handwritten kernel, when available, and numerical gradients. See the
//! `rir-kernels` tests.
//!
//! V1 supports kernels with two logical axes, one inner reduction or scan
//! axis, and reads indexed directly by axes (no gathers). Every unsupported
//! case produces an explicit error.

use crate::ids::IrId;
use std::collections::HashMap;

use crate::builder::KernelBuilder;
use crate::ir::*;
use crate::types::DType;
use crate::validate::ValidateError;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AutodiffError {
    /// V1 requires two logical axes.
    #[error("autodiff v1: expected two axes, {got} declared")]
    AxisCount { got: usize },
    /// V1 supports a single reduction or scan axis.
    #[error("autodiff v1: only one reduction/scan axis")]
    MultipleInnerAxes,
    #[error("%{}: no transpose for {what} - use a hand-written kernel",.value.0)]
    UnsupportedOp { value: ValueId, what: &'static str },
    /// A read or write index is not an axis `Index` (a gather).
    #[error("%{}: non-affine index (gather) - not transposed in v1",.value.0)]
    NonAffineIndex { value: ValueId },
    /// The same argument is read with two different index patterns.
    #[error("argument #{} read with different indices",.arg.0)]
    InconsistentReadIndex { arg: ArgId },
    /// An unread input provides no iteration space for writing its gradient.
    #[error("argument #{} never read: gradient without an iteration space",.arg.0)]
    UnusedInput { arg: ArgId },
    #[error("argument #{} is quantized: no gradient in v1",.arg.0)]
    QuantInput { arg: ArgId },
    #[error("invalid derived kernel: {0}")]
    Validate(#[from] ValidateError),
}

/// Backward builder plus inner-axis dependency tracking, which is required to
/// transpose broadcasts into reductions.
struct Ad {
    b: KernelBuilder,
    dep: HashMap<ValueId, bool>,
    raxis: Option<AxisId>,
    minus_one: Option<ValueId>,
    zero: Option<ValueId>,
    half: Option<ValueId>,
}

impl Ad {
    fn tag(&mut self, v: ValueId, d: bool) -> ValueId {
        self.dep.insert(v, d);
        v
    }

    fn d(&self, v: ValueId) -> bool {
        *self.dep.get(&v).unwrap_or(&false)
    }

    fn cst(&mut self, c: f32) -> ValueId {
        let v = self.b.const_f32(c);
        self.tag(v, false)
    }

    /// Uniform along every axis, so it is tagged independent like a constant.
    fn axis_extent(&mut self, axis: AxisId) -> ValueId {
        let v = self.b.axis_extent(axis);
        self.tag(v, false)
    }

    fn read(&mut self, arg: ArgId, axes: &[AxisId]) -> ValueId {
        let v = self.b.read(arg, axes);
        let d = self.raxis.is_some_and(|r| axes.contains(&r));
        self.tag(v, d)
    }

    fn add(&mut self, a: ValueId, bx: ValueId) -> ValueId {
        let d = self.d(a) || self.d(bx);
        let v = self.b.add(a, bx);
        self.tag(v, d)
    }

    fn sub(&mut self, a: ValueId, bx: ValueId) -> ValueId {
        let d = self.d(a) || self.d(bx);
        let v = self.b.sub(a, bx);
        self.tag(v, d)
    }

    fn mul(&mut self, a: ValueId, bx: ValueId) -> ValueId {
        let d = self.d(a) || self.d(bx);
        let v = self.b.mul(a, bx);
        self.tag(v, d)
    }

    fn div(&mut self, a: ValueId, bx: ValueId) -> ValueId {
        let d = self.d(a) || self.d(bx);
        let v = self.b.div(a, bx);
        self.tag(v, d)
    }

    fn sqrt(&mut self, a: ValueId) -> ValueId {
        let d = self.d(a);
        let v = self.b.sqrt(a);
        self.tag(v, d)
    }

    fn exp(&mut self, a: ValueId) -> ValueId {
        let d = self.d(a);
        let v = self.b.exp(a);
        self.tag(v, d)
    }

    fn tanh(&mut self, a: ValueId) -> ValueId {
        let d = self.d(a);
        let v = self.b.tanh(a);
        self.tag(v, d)
    }

    fn cmp(&mut self, op: CmpOp, a: ValueId, bx: ValueId) -> ValueId {
        let d = self.d(a) || self.d(bx);
        let v = self.b.cmp(op, a, bx);
        self.tag(v, d)
    }

    fn select(&mut self, c: ValueId, t: ValueId, f: ValueId) -> ValueId {
        let d = self.d(c) || self.d(t) || self.d(f);
        let v = self.b.select(c, t, f);
        self.tag(v, d)
    }

    fn reduce(&mut self, op: ReduceOp, axis: AxisId, v: ValueId) -> ValueId {
        let r = self
            .b
            .reduce(op, axis, v, ReductionSemantics::Deterministic);
        self.tag(r, false)
    }

    fn scan(&mut self, op: ScanOp, axis: AxisId, dir: ScanDirection, v: ValueId) -> ValueId {
        let d = self.raxis == Some(axis) || self.d(v);
        let s = self.b.scan(op, axis, dir, v);
        self.tag(s, d)
    }

    fn neg(&mut self, v: ValueId) -> ValueId {
        let m1 = match self.minus_one {
            Some(m) => m,
            None => {
                let m = self.cst(-1.0);
                self.minus_one = Some(m);
                m
            }
        };
        self.mul(v, m1)
    }

    fn zero(&mut self) -> ValueId {
        match self.zero {
            Some(z) => z,
            None => {
                let z = self.cst(0.0);
                self.zero = Some(z);
                z
            }
        }
    }

    fn half(&mut self) -> ValueId {
        match self.half {
            Some(h) => h,
            None => {
                let h = self.cst(0.5);
                self.half = Some(h);
                h
            }
        }
    }

    /// Applies broadcast transposition. If `v_uniform` is uniform along the
    /// forward inner axis but its adjoint depends on that axis, the total
    /// adjoint is their sum over the axis.
    fn collapse(&mut self, g: ValueId, v_uniform: bool) -> ValueId {
        match self.raxis {
            Some(r) if v_uniform && self.d(g) => self.reduce(ReduceOp::Sum, r, g),
            _ => g,
        }
    }
}

fn axes_of(k: &Kernel, at: ValueId, idx: &[ValueId]) -> Result<Vec<AxisId>, AutodiffError> {
    idx.iter()
        .map(|&v| match k.ops[v.0 as usize] {
            Op::Index(a) => Ok(a),
            _ => Err(AutodiffError::NonAffineIndex { value: at }),
        })
        .collect()
}

/// Copies a forward value into the backward kernel, with memoization.
fn import(
    ad: &mut Ad,
    k: &Kernel,
    pvals: &[ValueId],
    map: &mut HashMap<ValueId, ValueId>,
    v: ValueId,
) -> Result<ValueId, AutodiffError> {
    if let Some(&nv) = map.get(&v) {
        return Ok(nv);
    }
    let nv = match k.ops[v.0 as usize].clone() {
        Op::ConstF32(c) => ad.cst(c),
        // The backward kernel reuses the forward axis ids, so the extent
        // carries over unchanged.
        Op::AxisExtent(a) => ad.axis_extent(a),
        Op::Param(p) => pvals[p.0 as usize],
        Op::Index(_) => {
            // Index values only occur in Read/Write indices imported through
            // `axes_of`; encountering one as a value denotes a gather.
            return Err(AutodiffError::NonAffineIndex { value: v });
        }
        // The transpose of a repeat is a *reduction* over the replayed range,
        // which is a shape autodiff would have to introduce rather than
        // transpose. Refused with its reason rather than differentiated as if
        // the fold were the identity (ADR-1 section 5).
        Op::RepeatIndex { .. } => {
            return Err(AutodiffError::UnsupportedOp {
                value: v,
                what: "RepeatIndex",
            });
        }
        Op::Read { tensor, idx } => {
            let axes = axes_of(k, v, &idx)?;
            ad.read(tensor, &axes)
        }
        Op::Dequant { .. } => {
            return Err(AutodiffError::UnsupportedOp {
                value: v,
                what: "Dequant",
            });
        }
        Op::Add(a, b) => {
            let (a, b) = (import(ad, k, pvals, map, a)?, import(ad, k, pvals, map, b)?);
            ad.add(a, b)
        }
        Op::Sub(a, b) => {
            let (a, b) = (import(ad, k, pvals, map, a)?, import(ad, k, pvals, map, b)?);
            ad.sub(a, b)
        }
        Op::Mul(a, b) => {
            let (a, b) = (import(ad, k, pvals, map, a)?, import(ad, k, pvals, map, b)?);
            ad.mul(a, b)
        }
        Op::Div(a, b) => {
            let (a, b) = (import(ad, k, pvals, map, a)?, import(ad, k, pvals, map, b)?);
            ad.div(a, b)
        }
        Op::Sqrt(a) => {
            let a = import(ad, k, pvals, map, a)?;
            ad.sqrt(a)
        }
        Op::Exp(a) => {
            let a = import(ad, k, pvals, map, a)?;
            ad.exp(a)
        }
        Op::Tanh(a) => {
            let a = import(ad, k, pvals, map, a)?;
            ad.tanh(a)
        }
        Op::Cmp { op, lhs, rhs } => {
            let (l, r) = (
                import(ad, k, pvals, map, lhs)?,
                import(ad, k, pvals, map, rhs)?,
            );
            ad.cmp(op, l, r)
        }
        Op::Select { cond, t, f } => {
            let c = import(ad, k, pvals, map, cond)?;
            let t = import(ad, k, pvals, map, t)?;
            let f = import(ad, k, pvals, map, f)?;
            ad.select(c, t, f)
        }
        Op::Reduce {
            op, axis, value, ..
        } => {
            let x = import(ad, k, pvals, map, value)?;
            ad.reduce(op, axis, x)
        }
        Op::Scan {
            op,
            axis,
            dir,
            value,
        } => {
            let x = import(ad, k, pvals, map, value)?;
            ad.scan(op, axis, dir, x)
        }
        Op::Write { .. } => unreachable!("Write is not a value"),
    };
    map.insert(v, nv);
    Ok(nv)
}

/// Derives the backward kernel of `k` by transposition. The resulting kernel
/// is named `{name}_grad`. Its arguments are, in order, each forward input (or
/// `d<output>` for a forward output), followed by one `d<input>` output for
/// every forward input.
pub fn derive_backward(k: &Kernel) -> Result<crate::ValidatedKernel, AutodiffError> {
    if k.axes.len() != 2 {
        return Err(AutodiffError::AxisCount { got: k.axes.len() });
    }
    for (i, arg) in k.args.iter().enumerate() {
        if matches!(arg.ty.dtype, DType::Quant(_)) {
            return Err(AutodiffError::QuantInput { arg: ArgId::at(i) });
        }
    }

    // Find the single inner reduction or scan axis.
    let mut raxis: Option<AxisId> = None;
    for op in &k.ops {
        let a = match op {
            Op::Reduce { axis, .. } | Op::Scan { axis, .. } => Some(*axis),
            _ => None,
        };
        if let Some(a) = a {
            match raxis {
                None => raxis = Some(a),
                Some(r) if r != a => return Err(AutodiffError::MultipleInnerAxes),
                _ => {}
            }
        }
    }

    let mut ad = Ad {
        b: KernelBuilder::new(&format!("{}_grad", k.name)),
        dep: HashMap::new(),
        raxis,
        minus_one: None,
        zero: None,
        half: None,
    };

    // Preserve forward argument indices: inputs remain inputs, outputs become
    // incoming gradients, followed by the output gradients.
    let mut mirror_args = Vec::new();
    for arg in &k.args {
        let id = match arg.access {
            Access::Read => ad.b.input(&arg.name, arg.ty),
            Access::Write => ad.b.input(&format!("d{}", arg.name), arg.ty),
        };
        mirror_args.push(id);
    }
    let mut pvals = Vec::new();
    for p in &k.params {
        let v = ad.b.param(&p.name, p.ty);
        ad.dep.insert(v, false);
        pvals.push(v);
    }
    for a in &k.axes {
        // Extents refer to ArgIds, which are preserved by construction.
        ad.b.axis(&a.name, a.extent);
    }
    let mut douts: HashMap<ArgId, ArgId> = HashMap::new();
    for (i, arg) in k.args.iter().enumerate() {
        if arg.access == Access::Read {
            douts.insert(ArgId::at(i), ad.b.output(&format!("d{}", arg.name), arg.ty));
        }
    }

    let mut fwd_map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut contribs: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    let mut darg: HashMap<ArgId, (Vec<AxisId>, Vec<ValueId>)> = HashMap::new();

    for i in (0..k.ops.len()).rev() {
        let vid = ValueId::at(i);
        if let Op::Write { tensor, idx, value } = &k.ops[i] {
            let axes = axes_of(k, vid, idx)?;
            let g = ad.read(mirror_args[tensor.0 as usize], &axes);
            contribs.entry(*value).or_default().push(g);
            continue;
        }

        let Some(list) = contribs.remove(&vid) else {
            continue;
        };
        let mut g = list[0];
        for &c in &list[1..] {
            g = ad.add(g, c);
        }
        let v_uniform = raxis.is_none_or(|r| !depends_on_axis(k, vid, r));
        g = ad.collapse(g, v_uniform);

        match k.ops[i].clone() {
            // No gradient flows into a shape: an extent is a constant of the
            // dispatch, like a parameter.
            Op::ConstF32(_) | Op::Param(_) | Op::Index(_) | Op::AxisExtent(_) => {}
            Op::Cmp { .. } => {}
            Op::Read { tensor, idx } => {
                let axes = axes_of(k, vid, &idx)?;
                match darg.get_mut(&tensor) {
                    Some((prev, list)) => {
                        if *prev != axes {
                            return Err(AutodiffError::InconsistentReadIndex { arg: tensor });
                        }
                        list.push(g);
                    }
                    None => {
                        darg.insert(tensor, (axes, vec![g]));
                    }
                }
            }
            Op::Dequant { .. } => {
                return Err(AutodiffError::UnsupportedOp {
                    value: vid,
                    what: "Dequant",
                });
            }
            Op::RepeatIndex { .. } => {
                return Err(AutodiffError::UnsupportedOp {
                    value: vid,
                    what: "RepeatIndex",
                });
            }
            Op::Add(a, b) => {
                contribs.entry(a).or_default().push(g);
                contribs.entry(b).or_default().push(g);
            }
            Op::Sub(a, b) => {
                contribs.entry(a).or_default().push(g);
                let n = ad.neg(g);
                contribs.entry(b).or_default().push(n);
            }
            Op::Mul(a, b) => {
                let ib = import(&mut ad, k, &pvals, &mut fwd_map, b)?;
                let ia = import(&mut ad, k, &pvals, &mut fwd_map, a)?;
                let ga = ad.mul(g, ib);
                let gb = ad.mul(g, ia);
                contribs.entry(a).or_default().push(ga);
                contribs.entry(b).or_default().push(gb);
            }
            Op::Div(a, b) => {
                let ib = import(&mut ad, k, &pvals, &mut fwd_map, b)?;
                let ia = import(&mut ad, k, &pvals, &mut fwd_map, a)?;
                let ga = ad.div(g, ib);
                contribs.entry(a).or_default().push(ga);
                let num = ad.mul(g, ia);
                let bb = ad.mul(ib, ib);
                let q = ad.div(num, bb);
                let gb = ad.neg(q);
                contribs.entry(b).or_default().push(gb);
            }
            Op::Sqrt(a) => {
                // d√a = g · ½ / √a
                let s = import(&mut ad, k, &pvals, &mut fwd_map, vid)?;
                let h = ad.half();
                let gh = ad.mul(g, h);
                let ga = ad.div(gh, s);
                contribs.entry(a).or_default().push(ga);
            }
            Op::Exp(a) => {
                let e = import(&mut ad, k, &pvals, &mut fwd_map, vid)?;
                let ga = ad.mul(g, e);
                contribs.entry(a).or_default().push(ga);
            }
            // d tanh(a) = g · (1 − tanh(a)²). Written on the forward value the
            // graph already holds, like `Exp` above, rather than on `1/cosh²`:
            // the transposition then reuses the node instead of recomputing a
            // second transcendental.
            Op::Tanh(a) => {
                let t = import(&mut ad, k, &pvals, &mut fwd_map, vid)?;
                let tt = ad.mul(t, t);
                let one = ad.cst(1.0);
                let d = ad.sub(one, tt);
                let ga = ad.mul(g, d);
                contribs.entry(a).or_default().push(ga);
            }
            Op::Select { cond, t, f } => {
                let c = import(&mut ad, k, &pvals, &mut fwd_map, cond)?;
                let z = ad.zero();
                let gt = ad.select(c, g, z);
                let gf = ad.select(c, z, g);
                contribs.entry(t).or_default().push(gt);
                contribs.entry(f).or_default().push(gf);
            }
            Op::Reduce {
                op: ReduceOp::Sum,
                value,
                ..
            } => {
                // Transposing the broadcast distributes the uniform adjoint.
                contribs.entry(value).or_default().push(g);
            }
            Op::Reduce {
                op: ReduceOp::Max, ..
            } => {
                return Err(AutodiffError::UnsupportedOp {
                    value: vid,
                    what: "Reduce(Max)",
                });
            }
            Op::Scan {
                op: ScanOp::Sum,
                axis,
                dir,
                value,
            } => {
                // The transpose of an inclusive scan is the opposite scan.
                let gs = ad.scan(ScanOp::Sum, axis, dir.flipped(), g);
                contribs.entry(value).or_default().push(gs);
            }
            Op::Write { .. } => unreachable!("Write is not a value"),
        }
    }

    for (i, arg) in k.args.iter().enumerate() {
        if arg.access != Access::Read {
            continue;
        }
        let Some((axes, list)) = darg.remove(&ArgId::at(i)) else {
            return Err(AutodiffError::UnusedInput { arg: ArgId::at(i) });
        };
        let mut g = list[0];
        for &c in &list[1..] {
            g = ad.add(g, c);
        }
        // An input read uniformly along the inner axis (a per-row bias) has a
        // gradient equal to the sum of contributions.
        let read_uniform = raxis.is_none_or(|r| !axes.contains(&r));
        g = ad.collapse(g, read_uniform);
        ad.b.write(douts[&ArgId::at(i)], &axes, g);
    }

    Ok(ad.b.finish()?)
}

#[cfg(test)]
mod tests {
    //! What transposition refuses, and what it produces.
    //!
    //! `derive_backward` had no `cfg(test)` of its own: it was covered by the one
    //! kernel the registry derives with it (`l2_norm_fwd_grad`), whose gradient is
    //! checked numerically in `rir-kernels`. That proves the happy path on one
    //! graph and says nothing about the **refusals**, which are the whole reason
    //! this returns a `Result`: a gather, a quantized input, a second inner axis.
    //! A refusal that silently became a wrong derivation would pass every test
    //! there was.

    use super::*;
    use crate::layout::Layout;
    use crate::{Extent, KernelBuilder, QuantType, ReduceOp, ReductionSemantics, TensorType};

    /// The forward kernel the registry actually transposes: one reduction, an
    /// elementwise epilogue.
    fn forward() -> crate::ValidatedKernel {
        let mut k = KernelBuilder::new("probe_fwd");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let xv = k.read(x, &[col, row]);
        let sq = k.mul(xv, xv);
        let s = k.reduce(ReduceOp::Sum, col, sq, ReductionSemantics::Deterministic);
        let r = k.sqrt(s);
        let d = k.div(xv, r);
        k.write(y, &[col, row], d);
        k.finish().unwrap()
    }

    /// The derived kernel is **validated**, and it is the type that says so.
    ///
    /// `derive_backward` builds a graph inside the crate, where the fields of
    /// `Kernel` are visible; returning `ValidatedKernel` is what forces its last
    /// line through the builder. This is the test that would fail if a
    /// future path returned an unvalidated one.
    #[test]
    fn the_derived_kernel_is_validated_and_writes_one_gradient_per_input() {
        let fwd = forward();
        let back = derive_backward(&fwd).expect("transposition");
        // One incoming cotangent per output, one gradient per read input.
        let outputs = back
            .args()
            .iter()
            .filter(|a| a.access == crate::Access::Write)
            .count();
        assert_eq!(outputs, 1, "one gradient, for the single input x");
        assert_eq!(back.axes().len(), 2, "the axes of the forward kernel");
        assert!(
            back.name().contains("probe_fwd"),
            "the derived kernel names its forward: {}",
            back.name()
        );
    }

    /// A quantized input is refused, and named.
    ///
    /// Not a limitation of the transposition rules but of the *representation*: a
    /// gradient with respect to a quantized tensor has no form to be written in,
    /// so a derivation that produced one would be writing floats into a block
    /// layout.
    #[test]
    fn a_quantized_input_is_refused() {
        let mut k = KernelBuilder::new("probe_quant");
        let x = k.input(
            "x",
            TensorType {
                dtype: DType::Quant(QuantType::Q8_0),
                rank: 2,
                layout: Layout::Ggml,
            },
        );
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        // `read` on a quantized argument is the fused Read + Dequant pair.
        let xv = k.read(x, &[col, row]);
        let s = k.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
        k.write(y, &[col, row], s);
        let fwd = k.finish().unwrap();
        assert_eq!(
            derive_backward(&fwd).err(),
            Some(AutodiffError::QuantInput { arg: ArgId(0) })
        );
    }

    /// A kernel with a single axis is refused: V1 transposes a `(row, col)`
    /// space, and there is no second axis to carry the reduction.
    #[test]
    fn a_one_axis_kernel_is_refused() {
        let mut k = KernelBuilder::new("probe_1d");
        let x = k.input("x", TensorType::f32(1));
        let y = k.output("y", TensorType::f32(1));
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let xv = k.read(x, &[col]);
        let e = k.exp(xv);
        k.write(y, &[col], e);
        let fwd = k.finish().unwrap();
        assert_eq!(
            derive_backward(&fwd).err(),
            Some(AutodiffError::AxisCount { got: 1 })
        );
    }
}
