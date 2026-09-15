//! The `UNARY` family: 1.8% of
//! measured traffic on Qwen3.5, 0.9% on gemma-3-270m.
//!
//! **One canonical table and `build(variant)`, never an association invented
//! from a name.** `GGML_OP_UNARY` is not an op but a *port*,
//! twenty-two functions behind one `ggml_op`, distinguished by an integer in
//! `op_params`. Deriving the member from the kernel name would recreate the bug
//! family already met on `GGML_OP_L2_NORM_FWD`, only worse: here the wrong
//! member *exists*, compiles, and silently computes another function.
//!
//! **What the family required from RIR, its only real cost.**
//! The sub-op is a **claim**, like `src0`'s dtype on `out_prod`: a `GELU_ERF`
//! node is not a candidate for the `silu` kernel. Without this, selection,
//! which falls back to a pair's fallback variant when no shape rule matches,
//! would assign an arbitrary table member to *every* `UNARY` node - a false
//! positive that would pass every lane counter. Hence
//! `DomainRestriction::OpVariant`, a new rejection-taxonomy value, and its
//! published rejection.
//!
//! **What RIR writes, and what it leaves.** Fourteen of twenty-two members. The
//! remaining eight are not deferred for lack of time, and the last is not even
//! a vocabulary issue:
//!
//! - `SOFTPLUS` needs `log`, `GELU_ERF` needs `erf` - two new scalar Loop IR
//!   operations **and** an arm in each of three emitters, for two members absent
//!   from both censuses;
//! - `FLOOR`, `CEIL`, `ROUND`, and `TRUNC` need directed rounding, the same cost
//!   for the same absence of traffic;
//! - `XIELU` carries its own scalars in `op_params`, requiring one parameter row
//!   per member where the table has one for the entire family;
//! - `GELU` is written, exact, and **removed from the table by measurement**.
//!   On Metal/M1 the benchmark cannot judge this member: the *native* kernel
//!   yields NMSE of 1.00–1.14e-7 against a 1.0e-7 threshold and fails zero to
//!   two of four F32 cases depending on the draw. The cause precedes both
//!   kernels - the CPU reference uses `tanhf`, the GPU uses `precise::tanh`, and
//!   one ulp is enough over the sampled `[-3, 3]` range. A RIR kernel written
//!   term for term like native yields the same error; declaring it would make
//!   promotion of the *entire* family depend on chance. The reopening trigger
//!   is explicit: when native Metal passes this threshold reliably, or when
//!   `test-backend-ops` revises the `GELU` tolerance, restore the table row,
//!   the kernel itself will not need rewriting.
//!
//! Each is rejected *with its reason*, and each
//! reopens on an explicit trigger.
//!
//! **In-place is safe without any declaration**, for the elementwise-strip
//! reason: every invocation reads and writes the same index, so `dst`/`src0`
//! aliasing does not change the result.

use rir_core::{
    CmpOp, Constraint, DType, Extent, KernelBuilder, TensorType, ValidateError, ValidatedKernel,
    ValueId,
};

/// A family member. One entry per `ggml_unary_op` written by RIR.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unary {
    Abs,
    Sgn,
    Neg,
    Step,
    Tanh,
    Elu,
    Relu,
    Sigmoid,
    GeluQuick,
    Silu,
    Hardswish,
    Hardsigmoid,
    Exp,
    Expm1,
}

/// Canonical table in `ggml_unary_op` order - the same order as
/// `GGML_UNARY_OP_NAME`, so cross-reading both tables is reading, not
/// reconstruction.
pub fn variants() -> [Unary; 14] {
    [
        Unary::Abs,
        Unary::Sgn,
        Unary::Neg,
        Unary::Step,
        Unary::Tanh,
        Unary::Elu,
        Unary::Relu,
        Unary::Sigmoid,
        Unary::GeluQuick,
        Unary::Silu,
        Unary::Hardswish,
        Unary::Hardsigmoid,
        Unary::Exp,
        Unary::Expm1,
    ]
}

impl Unary {
    /// Kernel suffix, hence the generated directory and entrypoint suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            Unary::Abs => "abs",
            Unary::Sgn => "sgn",
            Unary::Neg => "neg",
            Unary::Step => "step",
            Unary::Tanh => "tanh",
            Unary::Elu => "elu",
            Unary::Relu => "relu",
            Unary::Sigmoid => "sigmoid",
            Unary::GeluQuick => "gelu_quick",
            Unary::Silu => "silu",
            Unary::Hardswish => "hardswish",
            Unary::Hardsigmoid => "hardsigmoid",
            Unary::Exp => "exp",
            Unary::Expm1 => "expm1",
        }
    }

    pub fn kernel_name(self) -> String {
        format!("unary_{}", self.suffix())
    }

    /// The `ggml_unary_op` member, written explicitly. This is the claim
    /// published by the registry and evaluated at dispatch; matching against
    /// the enum happens in the fork using `ggml_unary_op_name`, never a numeric
    /// value copied here.
    pub fn ggml_unary_op(self) -> &'static str {
        match self {
            Unary::Abs => "GGML_UNARY_OP_ABS",
            Unary::Sgn => "GGML_UNARY_OP_SGN",
            Unary::Neg => "GGML_UNARY_OP_NEG",
            Unary::Step => "GGML_UNARY_OP_STEP",
            Unary::Tanh => "GGML_UNARY_OP_TANH",
            Unary::Elu => "GGML_UNARY_OP_ELU",
            Unary::Relu => "GGML_UNARY_OP_RELU",
            Unary::Sigmoid => "GGML_UNARY_OP_SIGMOID",
            Unary::GeluQuick => "GGML_UNARY_OP_GELU_QUICK",
            Unary::Silu => "GGML_UNARY_OP_SILU",
            Unary::Hardswish => "GGML_UNARY_OP_HARDSWISH",
            Unary::Hardsigmoid => "GGML_UNARY_OP_HARDSIGMOID",
            Unary::Exp => "GGML_UNARY_OP_EXP",
            Unary::Expm1 => "GGML_UNARY_OP_EXPM1",
        }
    }
}

/// `ggml_gelu_quick_f32` (`ggml/src/ggml-cpu/vec.h`).
const GELU_QUICK_COEF: f32 = -1.702;

pub fn build(member: Unary) -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new(&member.kernel_name());

    let x = k.input("x", TensorType::f32(4));
    let dst = k.output("dst", TensorType::f32(4));

    // Same axis distribution as the elementwise strip: `col` first because it
    // is contiguous, thus determines coalescing and is the only axis that can
    // carry a vectorized read.
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let idx = [col, row, plane, batch];

    let v = k.read(x, &idx);

    // Each arm copies `ggml/src/ggml-cpu/unary-ops.cpp` term for term,
    // **including associativity**: `A·x·x` groups left and `A·(x·x)` is not the
    // same F32. This separates NMSE below the threshold from NMSE above it, as
    // the member removed from the table demonstrates.
    let zero = k.const_f32(0.0);
    let one = k.const_f32(1.0);
    let value: ValueId = match member {
        Unary::Abs => {
            let neg = k.sub(zero, v);
            let c = k.cmp(CmpOp::Gt, v, zero);
            k.select(c, v, neg)
        }
        Unary::Sgn => {
            let minus_one = k.const_f32(-1.0);
            let neg = k.cmp(CmpOp::Lt, v, zero);
            let lower = k.select(neg, minus_one, zero);
            let pos = k.cmp(CmpOp::Gt, v, zero);
            k.select(pos, one, lower)
        }
        Unary::Neg => k.sub(zero, v),
        Unary::Step => {
            let c = k.cmp(CmpOp::Gt, v, zero);
            k.select(c, one, zero)
        }
        Unary::Tanh => k.tanh(v),
        Unary::Elu => {
            // ggml calls `expm1f`; writing `exp(x) − 1` is the same function but
            // not the same last bit near zero. ggml's `EXPM1` member *is*
            // `expf(x) - 1.0f`, so that one is exact.
            let e = k.exp(v);
            let m1 = k.sub(e, one);
            let c = k.cmp(CmpOp::Gt, v, zero);
            k.select(c, v, m1)
        }
        Unary::Relu => {
            let c = k.cmp(CmpOp::Gt, v, zero);
            k.select(c, v, zero)
        }
        Unary::Sigmoid => {
            let nx = k.sub(zero, v);
            let e = k.exp(nx);
            let d = k.add(one, e);
            k.div(one, d)
        }
        Unary::GeluQuick => {
            let c = k.const_f32(GELU_QUICK_COEF);
            let cx = k.mul(c, v);
            let e = k.exp(cx);
            let d = k.add(one, e);
            let s = k.div(one, d);
            k.mul(v, s)
        }
        Unary::Silu => {
            let nx = k.sub(zero, v);
            let e = k.exp(nx);
            let d = k.add(one, e);
            k.div(v, d)
        }
        Unary::Hardswish | Unary::Hardsigmoid => {
            let three = k.const_f32(3.0);
            let six = k.const_f32(6.0);
            let t = k.add(v, three);
            let t = k.div(t, six);
            let above = k.cmp(CmpOp::Gt, t, zero);
            let lo = k.select(above, t, zero);
            let under = k.cmp(CmpOp::Lt, lo, one);
            let clamped = k.select(under, lo, one);
            if member == Unary::Hardswish {
                k.mul(v, clamped)
            } else {
                clamped
            }
        }
        Unary::Exp => k.exp(v),
        Unary::Expm1 => {
            let e = k.exp(v);
            k.sub(e, one)
        }
    };

    k.write(dst, &idx, value);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: x, max: 4 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    /// Analytical reference written from `ggml/src/ggml-cpu/unary-ops.cpp`, not
    /// from the kernel. `ELU` is the only deliberate difference - ggml uses
    /// `expm1f`, RIR uses `exp(x) − 1` - and the test expresses this with a
    /// tolerance rather than copying the kernel's choice.
    fn reference(member: Unary, x: f32) -> f32 {
        match member {
            Unary::Abs => x.abs(),
            Unary::Sgn => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Unary::Neg => -x,
            Unary::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Unary::Tanh => x.tanh(),
            Unary::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp_m1()
                }
            }
            Unary::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Unary::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Unary::GeluQuick => x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp())),
            Unary::Silu => x / (1.0 + (-x).exp()),
            Unary::Hardswish => x * 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Unary::Hardsigmoid => 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Unary::Exp => x.exp(),
            Unary::Expm1 => x.exp() - 1.0,
        }
    }

    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            *v = (u * 2.0 - 1.0) * scale;
        }
    }

    fn run_case(member: Unary, schedule: Schedule, ne: [usize; 4], gap: usize) {
        let kernel = build(member).unwrap();
        let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
        let (n_col, n_row, n_plane, n_batch) = (ne[0], ne[1], ne[2], ne[3]);

        let packed = |g: usize| {
            [
                4,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let len = |g: usize| n_col * n_row * n_plane * n_batch * g;

        let mut seed = 0x51c0_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
        let mut x = vec![0f32; len(gap)];
        // A narrow range: these are values where all fourteen members are
        // finite, thus the only ones where elementwise comparison is meaningful
        // for all. `test-backend-ops` extremes (±150) saturate `exp` on both
        // sides of the comparison.
        fill(&mut seed, &mut x, 3.0);

        let mut got = vec![0f32; len(1)];
        let shape = [n_col, n_row, n_plane, n_batch];
        let mut args = [
            BoundArg::In(TensorView {
                data: &x,
                shape,
                nb: packed(gap),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut got,
                shape,
                nb: packed(1),
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for bt in 0..n_batch {
            for p in 0..n_plane {
                for r in 0..n_row {
                    let dst_base = ((bt * n_plane + p) * n_row + r) * n_col;
                    let x_base = ((bt * n_plane * gap + p * gap) * n_row + r) * n_col;
                    for c in 0..n_col {
                        let e = reference(member, x[x_base + c]);
                        let g = got[dst_base + c];
                        let tol = 1e-6f32.max(1e-5 * e.abs());
                        assert!(
                            (g - e).abs() <= tol,
                            "{member:?} {:?} [{bt},{p},{r},{c}] : {g} vs {e}",
                            schedule.backend()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for member in variants() {
            for ne in [[1, 1, 1, 1], [7, 3, 1, 1], [64, 4, 2, 3], [33, 1, 1, 1]] {
                run_case(member, Schedule::cpu_serial(), ne, 1);
            }
        }
    }

    /// The vectorized lowering, the only one executed by the fork, over the row
    /// lengths that matter: a multiple of four, all three remainders, and a row
    /// shorter than the vector. This is also the only suite exercising
    /// `Cmp`/`Select` **on vector registers**, the capability this family
    /// required from emitters.
    #[test]
    fn parity_of_the_vectorized_lowering_including_its_tail() {
        for member in variants() {
            for schedule in [
                Schedule::vulkan_grid_vec4([256, 1, 1]),
                Schedule::metal_grid_vec4([256, 1, 1]),
            ] {
                for n_col in [1usize, 3, 4, 5, 33] {
                    run_case(member, schedule.clone(), [n_col, 3, 2, 2], 2);
                }
            }
        }
    }

    /// Values separated by branching arms that **never** arise from random
    /// sampling: exact zero for `SGN`, `STEP`, `RELU`, and `ELU`, and both
    /// `HARDSIGMOID` ramp boundaries (`x = −3` and `x = 3`), where `min` and
    /// `max` meet.
    #[test]
    fn the_branch_points_each_member_actually_has() {
        let probes = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
        for member in variants() {
            let kernel = build(member).unwrap();
            let lk = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();
            let n = probes.len();
            let mut got = vec![0f32; n];
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&probes, n, 1)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n, 1)),
            ];
            run(&lk, &mut args, &[]).unwrap();
            for (i, &p) in probes.iter().enumerate() {
                let e = reference(member, p);
                let tol = 1e-6f32.max(1e-5 * e.abs());
                assert!(
                    (got[i] - e).abs() <= tol,
                    "{member:?} at x={p}: {} vs {e}",
                    got[i]
                );
            }
        }
    }

    /// The member is table data, never derived from the name: both columns must
    /// remain aligned, and a `unary_silu` claiming `GGML_UNARY_OP_GELU` would
    /// compile silently.
    #[test]
    fn the_kernel_name_and_the_claimed_member_agree() {
        for member in variants() {
            let claimed = member
                .ggml_unary_op()
                .strip_prefix("GGML_UNARY_OP_")
                .unwrap()
                .to_lowercase();
            assert_eq!(
                member.kernel_name(),
                format!("unary_{claimed}"),
                "{member:?}: kernel name and claimed member disagree"
            );
        }
    }

    #[allow(dead_code)]
    mod generated_silu {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/unary_silu/cpu.rs"
        ));
    }

    /// The **generated** CPU, not the interpreter, on a strided shape.
    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (33usize, 7usize, 40usize);
        let len = stride * n_row;
        let mut seed = 0x51c0_2027u64;
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut x, 3.0);
        let nb = [4usize, 4 * stride, 4 * stride * n_row, 4 * stride * n_row];

        let mut got = vec![0f32; len];
        generated_silu::unary_silu(
            n_col,
            n_row,
            1,
            1,
            generated_silu::TensorRef { data: &x, nb },
            generated_silu::TensorRefMut { data: &mut got, nb },
        );
        for r in 0..n_row {
            for c in 0..n_col {
                let i = r * stride + c;
                let e = reference(Unary::Silu, x[i]);
                assert!(
                    (got[i] - e).abs() <= 1e-6f32.max(1e-5 * e.abs()),
                    "generated silu [{r},{c}]: {} vs {e}",
                    got[i]
                );
            }
        }
    }
}
