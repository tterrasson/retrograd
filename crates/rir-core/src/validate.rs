//! Level-one validation (ADR-1 section 3), without a GPU or execution.
//!
//! Checks SSA form, access permissions, index arity, output coverage, and the
//! **kind** of every value. Schedule-dependent checks, such as reduction
//! semantics versus strategy, live in `rir-lower`.
//!
//! Kinds are what keeps a well-formed graph from lowering into invalid source:
//! `Add(bool, f32)` or a `Select` on a float condition are SSA-correct and
//! nonsensical, and none of the CPU, GLSL, or MSL emitters would compile. They
//! are rejected here, once, rather than independently by every backend.

use crate::ids::IrId;

use crate::ir::*;
use crate::layout::MAX_TENSOR_RANK;
use crate::types::DType;

/// The kind of a value in the graph: what an operation may consume, and what
/// it produces. Deliberately coarser than a dtype - v1 computes in F32 - but
/// enough to separate the three things that must not mix: an index, a
/// predicate, and a number. A quantized read is its own kind, because only
/// `Dequant` may consume it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Index,
    F32,
    Bool,
    Quant,
    /// A `Write`, which occupies an SSA slot without producing a value.
    Unit,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Index => "index",
            Kind::F32 => "f32",
            Kind::Bool => "bool",
            Kind::Quant => "quantized",
            Kind::Unit => "without a value",
        }
    }
}

/// Checks an operand's kind. Callers have already run `check_operand`, so the
/// operand is defined and its kind known.
fn expect(
    kinds: &[Kind],
    value: ValueId,
    operand: ValueId,
    want: Kind,
) -> Result<(), ValidateError> {
    let got = kinds[operand.0 as usize];
    if got == want {
        Ok(())
    } else {
        Err(ValidateError::BadOperandKind {
            value,
            operand,
            expected: want.name(),
            got: got.name(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ValidateError {
    #[error("value %{} used by %{} before its definition",.operand.0,.value.0)]
    UseBeforeDef { value: ValueId, operand: ValueId },
    #[error("%{} reads output argument #{}",.value.0,.arg.0)]
    ReadFromOutput { value: ValueId, arg: ArgId },
    #[error("%{} writes to input argument #{}",.value.0,.arg.0)]
    WriteToInput { value: ValueId, arg: ArgId },
    #[error("%{} indexes argument #{} with {got} indices (rank {expected})",.value.0,.arg.0)]
    IndexArityMismatch {
        value: ValueId,
        arg: ArgId,
        expected: usize,
        got: usize,
    },
    #[error("output argument #{} is never written",.arg.0)]
    OutputNeverWritten { arg: ArgId },
    /// Inputs may be F32 or quantized; outputs must be F32 in v1.
    #[error("unsupported dtype {} (argument #{})",.dtype.name(),.arg.0)]
    UnsupportedDType { arg: ArgId, dtype: DType },
    /// `Dequant` must directly consume a `Read` from a tensor with the same
    /// quantized format.
    #[error("%{}: Dequant without a direct quantized Read",.value.0)]
    MalformedDequant { value: ValueId },
    #[error("%{} references unknown argument #{}",.value.0,.arg.0)]
    UnknownArg { value: ValueId, arg: ArgId },
    #[error("%{} references unknown axis #{}",.value.0,.axis.0)]
    UnknownAxis { value: ValueId, axis: AxisId },
    #[error("%{} references unknown parameter #{}",.value.0,.param.0)]
    UnknownParam { value: ValueId, param: ParamId },
    /// An operand's kind is not the one the operation consumes.
    #[error("%{} expects a {expected} operand, %{} is {got}",.value.0,.operand.0)]
    BadOperandKind {
        value: ValueId,
        operand: ValueId,
        expected: &'static str,
        got: &'static str,
    },
    /// A malformed `RepeatIndex` (ADR-1 section 5). Three shapes, one
    /// cause: the fold would produce a divisor the contract cannot publish.
    ///
    /// - `over` is not declared on the dimension being folded - the divisor
    ///   must be the repeated argument's own extent there, so that `arg_axes`
    ///   publishes it and a dispatcher fills it;
    /// - `over` is the very axis being folded - the identity written as a
    ///   modulo, a division per element for nothing;
    /// - `index` is not a direct axis position - a fold of a fold has no
    ///   relation `supports_op` or the registry can express.
    #[error(
        "%{}: repeated index on axis #{} - the `over` axis must carry \
         the extent of the repeated dimension and be distinct from it",
        .value.0, .axis.0
    )]
    MalformedRepeat { value: ValueId, axis: AxisId },
    /// An axis extent naming an argument outside the table, or a dimension
    /// outside that argument's rank. `supports_op` resolves every extent
    /// against the caller's `TensorDesc`, so an unresolvable one is a panic
    /// there and never a rejection.
    #[error(
        "axis #{} takes its extent from dimension {dim} of argument #{}, \
         which no argument of that rank has",
        .axis.0, .arg.0
    )]
    UnknownExtent {
        axis: AxisId,
        arg: ArgId,
        dim: usize,
    },
    /// A contract clause naming an argument outside the table - same reading
    /// as `UnknownExtent`: `supports_op` indexes the descriptors directly.
    #[error("a constraint references unknown argument #{}",.arg.0)]
    UnknownConstraintArg { arg: ArgId },
    /// A rank no ggml tensor has. `ne[]` and `nb[]` hold four dimensions, and
    /// everything that walks an argument's dimensions - `arg_axes`, the axis
    /// agreement of `supports_op` - indexes them with this rank.
    #[error(
        "argument #{} has rank {rank}: a ggml tensor has at most {MAX_TENSOR_RANK}",
        .arg.0
    )]
    UnsupportedRank { arg: ArgId, rank: usize },
}

/// A kernel `validate` has accepted - the value the rest of the pipeline takes.
///
/// It is a **witness**, not a second set of checks. `rir_lower::lower` indexes
/// `ops`, `axes` and `args` directly and therefore accepts only this validated
/// wrapper. The only constructor is
/// [`KernelBuilder::finish`](crate::KernelBuilder::finish), which validates the
/// kernel before it enters the pipeline.
///
/// It derefs to `&Kernel`, so everything that reads a kernel - `arg_axes`,
/// `supports_op`, the schedule table, the emitters - keeps taking `&Kernel` and
/// receives this by coercion. The coercion runs one way only, which is the
/// whole point.
///
/// A kernel assembled field by field does not compile, which is the half of the
/// guarantee no runtime test can express:
///
/// ```compile_fail
/// // `Kernel`'s fields are crate-private.
/// let k = rir_core::Kernel {
///     name: "forged".to_string(),
///     args: Vec::new(),
///     params: Vec::new(),
///     axes: Vec::new(),
///     ops: Vec::new(),
///     constraints: Vec::new(),
/// };
/// ```
///
/// And the builder is the way in:
///
/// ```
/// use rir_core::{Extent, KernelBuilder, TensorType};
///
/// let mut b = KernelBuilder::new("copy");
/// let x = b.input("x", TensorType::f32_2d());
/// let y = b.output("y", TensorType::f32_2d());
/// let row = b.axis("row", Extent::Dim { arg: x, dim: 1 });
/// let col = b.axis("col", Extent::Dim { arg: x, dim: 0 });
/// let v = b.read(x, &[col, row]);
/// b.write(y, &[col, row], v);
/// let k = b.finish().expect("valid");
/// assert_eq!(k.name(), "copy");
/// assert_eq!(k.args().len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct ValidatedKernel(Kernel);

impl ValidatedKernel {
    /// Validates `k` and keeps the witness. Crate-private: outside this crate
    /// the only entry is `KernelBuilder::finish`, and `derive_backward` goes
    /// through the builder too.
    pub(crate) fn new(k: Kernel) -> Result<Self, ValidateError> {
        validate(&k)?;
        Ok(Self(k))
    }

    /// The kernel itself. `Deref` covers most uses; this is for the places that
    /// need the borrow written out - a closure's parameter type, a `map`.
    pub fn kernel(&self) -> &Kernel {
        &self.0
    }
}

impl std::ops::Deref for ValidatedKernel {
    type Target = Kernel;

    fn deref(&self) -> &Kernel {
        &self.0
    }
}

pub fn validate(k: &Kernel) -> Result<(), ValidateError> {
    for arg_idx in 0..k.args.len() {
        let arg = &k.args[arg_idx];
        // F16 is readable **and** writable since F4 (ADR-3 section 6): it
        // is a memory element type, and lowering narrows at the store the way
        // it already widened at the load. `BF16`, `I32`, `U32` and `Bool` stay
        // out - they are declarable in `DType` and no emitter prints them, and
        // a kernel that asked for one would address bytes that are not what it
        // thinks they are.
        let ok = matches!(
            (arg.access, arg.ty.dtype),
            (Access::Read, DType::F32 | DType::F16 | DType::Quant(_))
                | (Access::Write, DType::F32 | DType::F16)
        );
        if !ok {
            return Err(ValidateError::UnsupportedDType {
                arg: ArgId::at(arg_idx),
                dtype: arg.ty.dtype,
            });
        }
        if arg.ty.rank > MAX_TENSOR_RANK {
            return Err(ValidateError::UnsupportedRank {
                arg: ArgId::at(arg_idx),
                rank: arg.ty.rank,
            });
        }
    }

    // The axes and the constraints, before a single op is read: both are handed
    // to the builder as whole values - `axis` takes an `Extent`, `constrain` a
    // `Constraint`, and `ArgId` is publicly constructible - so their argument
    // references are unchecked at this boundary. `supports_op`
    // resolves them by indexing `descs` directly (supports.rs), which is a
    // panic on the dispatch path and not a rejection: the witness
    // `ValidatedKernel` publishes has to cover them.
    for (i, axis) in k.axes.iter().enumerate() {
        let Extent::Dim { arg, dim } = axis.extent;
        let rank = k.args.get(arg.0 as usize).map_or(0, |a| a.ty.rank);
        if arg.0 as usize >= k.args.len() || dim >= rank {
            return Err(ValidateError::UnknownExtent {
                axis: AxisId::at(i),
                arg,
                dim,
            });
        }
    }
    for c in &k.constraints {
        let (Constraint::DType { arg, .. }
        | Constraint::Rank { arg, .. }
        | Constraint::Contiguous { arg }) = c;
        if arg.0 as usize >= k.args.len() {
            return Err(ValidateError::UnknownConstraintArg { arg: *arg });
        }
    }

    let mut written = vec![false; k.args.len()];
    // `kinds[i]` is the kind of `ValueId(i)`; operands always precede their
    // users, so one forward pass suffices.
    let mut kinds: Vec<Kind> = Vec::with_capacity(k.ops.len());

    for (i, op) in k.ops.iter().enumerate() {
        let value = ValueId::at(i);
        let check_operand = |operand: ValueId| -> Result<(), ValidateError> {
            if operand.0 as usize >= i {
                Err(ValidateError::UseBeforeDef { value, operand })
            } else {
                Ok(())
            }
        };
        let check_arg = |arg: ArgId| -> Result<(), ValidateError> {
            if arg.0 as usize >= k.args.len() {
                Err(ValidateError::UnknownArg { value, arg })
            } else {
                Ok(())
            }
        };
        let check_idx = |arg: ArgId, idx: &[ValueId]| -> Result<(), ValidateError> {
            let expected = k.args[arg.0 as usize].ty.rank;
            if idx.len() != expected {
                Err(ValidateError::IndexArityMismatch {
                    value,
                    arg,
                    expected,
                    got: idx.len(),
                })
            } else {
                Ok(())
            }
        };

        let kind = match op {
            Op::ConstF32(_) => Kind::F32,
            Op::Index(axis) => {
                if axis.0 as usize >= k.axes.len() {
                    return Err(ValidateError::UnknownAxis { value, axis: *axis });
                }
                Kind::Index
            }
            // Same axis check as `Index`, different kind: a position is an
            // index, an extent is a number the arithmetic ops may consume.
            Op::AxisExtent(axis) => {
                if axis.0 as usize >= k.axes.len() {
                    return Err(ValidateError::UnknownAxis { value, axis: *axis });
                }
                Kind::F32
            }
            Op::Param(p) => {
                if p.0 as usize >= k.params.len() {
                    return Err(ValidateError::UnknownParam { value, param: *p });
                }
                Kind::F32
            }
            // `index % extent(over)`: an index in, an index out. What is checked
            // here is only what is local - the axes exist and differ; that
            // `over` really belongs to the dimension it folds is checked below,
            // where the `Read` says which dimension that is.
            //
            // `index` must be a **direct** `Op::Index`, and that is a
            // restriction rather than an oversight. `Kind::Index` alone would
            // also admit another `RepeatIndex`, i.e. a fold of a fold - which
            // the rest of the pipeline cannot describe: `supports_op` and
            // `LoopKernel::folds` both publish a relation only when the operand
            // is an axis position, so a chain would validate, lower, and then
            // reach a dispatcher with a divisibility nobody checks and an
            // extent nobody fills. Refusing it here is the one place where the
            // refusal is cheap and total (ADR-1 section 5).
            Op::RepeatIndex { index, over } => {
                check_operand(*index)?;
                expect(&kinds, value, *index, Kind::Index)?;
                if over.0 as usize >= k.axes.len() {
                    return Err(ValidateError::UnknownAxis { value, axis: *over });
                }
                match k.ops[index.0 as usize] {
                    Op::Index(a) if a != *over => {}
                    _ => return Err(ValidateError::MalformedRepeat { value, axis: *over }),
                }
                Kind::Index
            }
            Op::Read { tensor, idx } => {
                check_arg(*tensor)?;
                if k.args[tensor.0 as usize].access != Access::Read {
                    return Err(ValidateError::ReadFromOutput {
                        value,
                        arg: *tensor,
                    });
                }
                check_idx(*tensor, idx)?;
                for (d, &v) in idx.iter().enumerate() {
                    check_operand(v)?;
                    expect(&kinds, value, v, Kind::Index)?;
                    // The divisor of a folded dimension is that dimension's own
                    // extent, on **this** argument. Anything else and the
                    // shader would divide by a number no one filled with the
                    // right `ne`.
                    if let Op::RepeatIndex { over, .. } = k.ops[v.0 as usize] {
                        let Extent::Dim { arg, dim } = k.axes[over.0 as usize].extent;
                        if arg != *tensor || dim != d {
                            return Err(ValidateError::MalformedRepeat { value, axis: over });
                        }
                    }
                }
                match k.args[tensor.0 as usize].ty.dtype {
                    DType::Quant(_) => Kind::Quant,
                    _ => Kind::F32,
                }
            }
            Op::Write {
                tensor,
                idx,
                value: v,
            } => {
                check_arg(*tensor)?;
                if k.args[tensor.0 as usize].access != Access::Write {
                    return Err(ValidateError::WriteToInput {
                        value,
                        arg: *tensor,
                    });
                }
                check_idx(*tensor, idx)?;
                for &iv in idx {
                    check_operand(iv)?;
                    expect(&kinds, value, iv, Kind::Index)?;
                }
                check_operand(*v)?;
                expect(&kinds, value, *v, Kind::F32)?;
                written[tensor.0 as usize] = true;
                Kind::Unit
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
                check_operand(*a)?;
                check_operand(*b)?;
                expect(&kinds, value, *a, Kind::F32)?;
                expect(&kinds, value, *b, Kind::F32)?;
                Kind::F32
            }
            Op::Sqrt(a) | Op::Exp(a) | Op::Tanh(a) => {
                check_operand(*a)?;
                expect(&kinds, value, *a, Kind::F32)?;
                Kind::F32
            }
            Op::Cmp { lhs, rhs, .. } => {
                check_operand(*lhs)?;
                check_operand(*rhs)?;
                expect(&kinds, value, *lhs, Kind::F32)?;
                expect(&kinds, value, *rhs, Kind::F32)?;
                Kind::Bool
            }
            Op::Select { cond, t, f } => {
                check_operand(*cond)?;
                check_operand(*t)?;
                check_operand(*f)?;
                expect(&kinds, value, *cond, Kind::Bool)?;
                expect(&kinds, value, *t, Kind::F32)?;
                expect(&kinds, value, *f, Kind::F32)?;
                Kind::F32
            }
            Op::Dequant { value: v, from } => {
                check_operand(*v)?;
                expect(&kinds, value, *v, Kind::Quant)?;
                let ok = matches!(
                    &k.ops[v.0 as usize],
                    Op::Read { tensor, .. }
                        if k.args[tensor.0 as usize].ty.dtype == DType::Quant(*from)
                );
                if !ok {
                    return Err(ValidateError::MalformedDequant { value });
                }
                Kind::F32
            }
            Op::Reduce { axis, value: v, .. } | Op::Scan { axis, value: v, .. } => {
                if axis.0 as usize >= k.axes.len() {
                    return Err(ValidateError::UnknownAxis { value, axis: *axis });
                }
                check_operand(*v)?;
                expect(&kinds, value, *v, Kind::F32)?;
                Kind::F32
            }
        };
        kinds.push(kind);
    }

    for (i, arg) in k.args.iter().enumerate() {
        if arg.access == Access::Write && !written[i] {
            return Err(ValidateError::OutputNeverWritten { arg: ArgId::at(i) });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::KernelBuilder;
    use crate::layout::TensorType;

    #[test]
    fn an_elementwise_copy_validates() {
        let mut k = KernelBuilder::new("copy");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        k.write(y, &[col, row], v);
        assert!(k.finish().is_ok());
    }

    /// The three references the builder takes as whole values, none of which
    /// passes through an op. Invalid references must be rejected before
    /// `supports_op` indexes the caller's descriptors with them.
    #[test]
    fn a_reference_outside_the_argument_table_is_refused() {
        let base = || {
            let mut k = KernelBuilder::new("bad");
            let x = k.input("x", TensorType::f32_2d());
            let y = k.output("y", TensorType::f32_2d());
            (k, x, y)
        };
        let finish = |mut k: KernelBuilder, x: ArgId, y: ArgId| {
            let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
            let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
            let v = k.read(x, &[col, row]);
            k.write(y, &[col, row], v);
            k.finish()
        };

        // An extent on an argument that does not exist.
        let (mut k, x, y) = base();
        k.axis(
            "ghost",
            Extent::Dim {
                arg: ArgId::at(999),
                dim: 0,
            },
        );
        assert!(matches!(
            finish(k, x, y),
            Err(ValidateError::UnknownExtent { .. })
        ));

        // An extent on a dimension the argument does not have: rank 2 here,
        // and `ne[src_dim]` would read past the descriptor.
        let (mut k, x, y) = base();
        k.axis("past", Extent::Dim { arg: x, dim: 3 });
        assert!(matches!(
            finish(k, x, y),
            Err(ValidateError::UnknownExtent { .. })
        ));

        // A contract clause on an argument that does not exist.
        let (mut k, x, y) = base();
        k.constrain(Constraint::Contiguous {
            arg: ArgId::at(999),
        });
        assert!(matches!(
            finish(k, x, y),
            Err(ValidateError::UnknownConstraintArg { .. })
        ));
    }

    /// A rank ggml has no room for. `arg_axes` builds one slot per dimension
    /// and `supports_op` reads `ne[d]` for each - four wide, both of them.
    #[test]
    fn a_rank_above_four_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32(5));
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row, col, row, col]);
        k.write(y, &[col, row], v);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::UnsupportedRank { rank: 5, .. })
        ));
    }

    #[test]
    fn writing_to_an_input_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let _y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        k.write(x, &[col, row], v);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::WriteToInput { .. })
        ));
    }

    #[test]
    fn an_output_never_written_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let _y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let _v = k.read(x, &[col, row]);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::OutputNeverWritten { .. })
        ));
    }

    #[test]
    fn a_wrong_index_arity_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col]);
        k.write(y, &[col, row], v);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::IndexArityMismatch { .. })
        ));
    }

    /// `Add(bool, f32)` is SSA-correct and would emit source that no backend
    /// compiles: it is a validation error, not a compilation one.
    #[test]
    fn arithmetic_on_a_boolean_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        let zero = k.const_f32(0.0);
        let cond = k.gt(v, zero);
        let bad = k.add(cond, v);
        k.write(y, &[col, row], bad);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::BadOperandKind {
                expected: "f32",
                got: "bool",
                ..
            })
        ));
    }

    /// A `Select` whose condition is a float, symmetrically.
    #[test]
    fn a_select_on_a_float_condition_is_refused() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        let zero = k.const_f32(0.0);
        let bad = k.select(v, v, zero);
        k.write(y, &[col, row], bad);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::BadOperandKind {
                expected: "bool",
                got: "f32",
                ..
            })
        ));
    }

    /// An index is not a number: using a loop position as an operand would
    /// need an explicit cast the IR does not have yet.
    #[test]
    fn an_index_is_not_a_number() {
        let mut k = KernelBuilder::new("bad");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        // `Op::Index(col)` is `ValueId(0)`: the builder cached it for `read`.
        let bad = k.mul(ValueId(0), v);
        k.write(y, &[col, row], bad);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::BadOperandKind {
                expected: "f32",
                got: "index",
                ..
            })
        ));
    }

    #[test]
    fn a_dtype_without_a_memory_access_type_is_refused() {
        let mut k = KernelBuilder::new("bad");
        // `BF16` and not `F16`: F16 became a supported element type in F4, and
        // what this test guards is the *rule*, not the list - a dtype the
        // emitters cannot print must fail at validation and not at emission.
        let ty = TensorType {
            dtype: DType::BF16,
            rank: 2,
            layout: crate::layout::Layout::Ggml,
        };
        let x = k.input("x", ty);
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let v = k.read(x, &[col, row]);
        k.write(y, &[col, row], v);
        assert!(matches!(
            k.finish(),
            Err(ValidateError::UnsupportedDType { .. })
        ));
    }

    /// The three malformed `RepeatIndex` forms (ADR-1 section 5). Each
    /// would produce a divisor the contract cannot publish, and the third - a
    /// repetition of a repetition - is the one that `Kind::Index` alone let
    /// through: it validated, lowered, and reached dispatch with divisibility
    /// that nobody checks and an extent that nobody fills.
    ///
    /// The graph is built by hand rather than through `KernelBuilder`, because
    /// the builder cannot produce any of these three forms - precisely why
    /// validation must reject them: it is the only barrier for IR built another
    /// way.
    ///
    /// Here "another way" means *inside this crate*:
    /// the fields are crate-private, so no caller can present validation with
    /// one of these graphs. The test stays because `validate` is what makes the
    /// builder's guarantee true, and because the builder is not the only writer
    /// here - `autodiff` assembles a graph through it and `Kind::Index` alone
    /// would let the third form pass.
    #[test]
    fn a_malformed_repeat_is_refused_before_lowering() {
        use crate::ir::{Access, Arg, AxisDecl, Extent};
        use crate::layout::TensorType;

        let kernel = |extra: &dyn Fn(&mut Vec<Op>) -> ValueId| -> Result<(), ValidateError> {
            let b = ArgId(0);
            let dst = ArgId(1);
            let (col, row, col_b) = (AxisId(0), AxisId(1), AxisId(2));
            let mut ops = vec![Op::Index(col), Op::Index(row)];
            let rep = extra(&mut ops);
            let read = ValueId::at(ops.len());
            ops.push(Op::Read {
                tensor: b,
                idx: vec![rep, ValueId(1)],
            });
            ops.push(Op::Write {
                tensor: dst,
                idx: vec![ValueId(0), ValueId(1)],
                value: read,
            });
            let k = Kernel {
                name: "rep".into(),
                args: vec![
                    Arg {
                        name: "b".into(),
                        ty: TensorType::f32_2d(),
                        access: Access::Read,
                    },
                    Arg {
                        name: "dst".into(),
                        ty: TensorType::f32_2d(),
                        access: Access::Write,
                    },
                ],
                params: vec![],
                axes: vec![
                    AxisDecl {
                        name: "col".into(),
                        extent: Extent::Dim { arg: b, dim: 0 },
                    },
                    AxisDecl {
                        name: "row".into(),
                        extent: Extent::Dim { arg: b, dim: 1 },
                    },
                    AxisDecl {
                        name: "col_b".into(),
                        extent: Extent::Dim { arg: b, dim: 0 },
                    },
                ],
                ops,
                constraints: vec![],
            };
            let _ = (col, row, col_b);
            validate(&k)
        };

        // The valid case: `col` repeated over the extent of dimension 0 of `b`.
        assert_eq!(
            kernel(&|ops: &mut Vec<Op>| {
                let v = ValueId::at(ops.len());
                ops.push(Op::RepeatIndex {
                    index: ValueId(0),
                    over: AxisId(2),
                });
                v
            }),
            Ok(())
        );
        // `over` is not declared on the repeated dimension: `row` carries the
        // extent of dimension 1, and the shader would divide by it.
        assert!(matches!(
            kernel(&|ops: &mut Vec<Op>| {
                let v = ValueId::at(ops.len());
                ops.push(Op::RepeatIndex {
                    index: ValueId(0),
                    over: AxisId(1),
                });
                v
            }),
            Err(ValidateError::MalformedRepeat { .. })
        ));
        // A repetition **of a repetition**: the operand is not an axis position,
        // so neither `supports_op` nor the registry has a relation to publish.
        assert!(matches!(
            kernel(&|ops: &mut Vec<Op>| {
                let inner = ValueId::at(ops.len());
                ops.push(Op::RepeatIndex {
                    index: ValueId(0),
                    over: AxisId(2),
                });
                let v = ValueId::at(ops.len());
                ops.push(Op::RepeatIndex {
                    index: inner,
                    over: AxisId(2),
                });
                v
            }),
            Err(ValidateError::MalformedRepeat { .. })
        ));
    }
}
