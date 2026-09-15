//! Loop IR expressions: what a register may be computed from.

use rir_core::{ArgId, AxisId, CmpOp, ParamId, ReduceOp};

use super::*;

#[derive(Clone, Debug)]
pub enum LExpr {
    ConstF32(f32),
    Param(ParamId),
    /// Extent of an axis as an F32 register. Every emitter already prints that
    /// extent as a loop bound, so this reuses the same push constant rather
    /// than introducing a new one.
    AxisExtent(AxisId),
    Copy(VarId),
    Add(VarId, VarId),
    Sub(VarId, VarId),
    Mul(VarId, VarId),
    Div(VarId, VarId),
    Sqrt(VarId),
    Exp(VarId),
    Tanh(VarId),
    Cmp {
        op: CmpOp,
        lhs: VarId,
        rhs: VarId,
    },
    Select {
        cond: VarId,
        t: VarId,
        f: VarId,
    },
    /// Integer division by a constant (`Idx` registers), used for blocking.
    IDivC(VarId, u32),
    /// Integer modulo by a constant (`Idx` registers).
    IModC(VarId, u32),
    /// Integer multiplication by a constant (`Idx` registers). With `IDivC` it
    /// is what rounds an index down to its tile origin, and with `IAdd` what
    /// turns a `(depth, row)` pair into an offset in a staged tile.
    IMulC(VarId, u32),
    /// Integer addition (`Idx` registers). Distinct from `Add`, which is the
    /// F32 one: the two banks are separate all the way to the emitters.
    IAdd(VarId, VarId),
    /// `x & c` on the `Idx` bank (ADR-3 section 4).
    ///
    /// Not a synonym of `IModC` with a power of two, and the distinction is the
    /// point: `IModC` is index arithmetic - the remainder of a division that
    /// means something - and this is a **bit field**. Writing `x & 15` as
    /// `IModC(x, 16)` was correct and unreadable the moment the field stopped
    /// starting at bit zero, which is every K quant.
    IAndC(VarId, u32),
    /// `x >> c` on the `Idx` bank. Same split as above against `IDivC`.
    IShrC(VarId, u32),
    /// `x >> s` with a **runtime** shift, both on the `Idx` bank.
    ///
    /// The index plan of a format gives the shift of element `e` as
    /// `bits · ((e % group) / plane)`, which is a register and not a constant.
    /// Without it a two-bit payload would need a four-way `Select` chain per
    /// element - the contortion this operator removes.
    IShr(VarId, VarId),
    /// `x | y` on the `Idx` bank, for concatenating disjoint bit fields.
    IOr(VarId, VarId),
    /// `x % extent(axis)` on the `Idx` bank: the fold of `Op::RepeatIndex`
    /// (ADR-1 section 5).
    ///
    /// Distinct from `IModC` because the divisor is a **runtime** extent, and
    /// distinct from a plain `IMod` because that extent is one every emitter
    /// already prints as a loop bound - so this reuses the push constant rather
    /// than introducing one, the same argument `LExpr::AxisExtent` makes for
    /// the F32 bank.
    IModAxis {
        var: VarId,
        axis: AxisId,
    },
    /// `LUT[q]` as an **F32** register: the table is printed as floats, so no
    /// emitter has to carry a signed integer table and no `Idx` register ever
    /// holds a negative value.
    Lut {
        table: rir_core::LutId,
        idx: VarId,
    },
    /// `Idx` register to F32. The conversion is explicit so that no emitter has
    /// to decide whether an integer register is a value or an address.
    IToF(VarId),
    /// An integer literal in an `Idx` register.
    ///
    /// The `Idx` bank had no literal because every index came from a builtin, a
    /// loop variable or an address term. Lowering an algorithm needs one: a
    /// Blelloch tree writes the identity at slot `lanes - 1`, and that slot is a
    /// number, not a coordinate.
    ConstIdx(u32),
    /// `x + c` on the `Idx` bank. The counterpart of `IMulC` for the offset
    /// half of an affine index - `lane · 2s + (2s - 1)` is one multiply and one
    /// add, both by constants the schedule fixes.
    IAddC(VarId, u32),
    /// `x - c` on the `Idx` bank. Written as its own variant rather than as
    /// `IAddC` with a wrapped constant: these are unsigned registers, and a
    /// subtraction that could underflow is a subtraction a reader must be able
    /// to see.
    ISubC(VarId, u32),
    /// `x - y` on the `Idx` bank, for the mirrored index of a backward scan
    /// (`n - 1 - p`).
    ISub(VarId, VarId),
    /// `x <op> c` on the `Idx` bank, into a `Bool` register. `LExpr::Cmp` is the
    /// F32 comparison; index comparisons are a different bank all the way to
    /// the emitters, and the guard of a tree level (`idx < lanes`) is one.
    ICmpC {
        op: CmpOp,
        var: VarId,
        c: u32,
    },
    /// The reduction combiner as an **expression**: `a + b` or `max(a, b)`.
    ///
    /// `Stmt::Accum` is the same operator applied to an accumulator; this is
    /// the form an algorithm needs when it combines two values it already holds
    /// and names the result - every level of a scan tree does exactly that.
    /// Both emitters already own the spelling (`combine`), so this costs them a
    /// call and no decision.
    Combine {
        op: ReduceOp,
        lhs: VarId,
        rhs: VarId,
    },
    /// Extent of an axis as an `Idx` register - the integer twin of
    /// `AxisExtent`, and the same argument for its existence: every emitter
    /// already prints that extent as a loop bound, so this reuses the push
    /// constant instead of introducing one.
    AxisExtentIdx(AxisId),
    /// Byte-address sum, in a register: the same `Σ terms` an access carries,
    /// evaluated once and named.
    ///
    /// It exists so that the **invariant part** of an address can leave a loop
    /// (`crate::hoist`). An access whose innermost term walks the loop keeps
    /// only that term plus this register, and the stride products of the outer
    /// axes - three multiplies and three adds per element on a rank-4 kernel,
    /// are paid once per loop instead of once per element.
    ///
    /// Deliberately the same `AddrTerm` list an access carries, rather than an
    /// integer expression tree: every emitter already prints that list, so
    /// hoisting costs each of them one `match` arm and no new algebra.
    AddrSum {
        arg: ArgId,
        terms: Vec<AddrTerm>,
    },
}
