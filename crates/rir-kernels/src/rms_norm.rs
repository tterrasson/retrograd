//! `rms_norm` - the **forward** RMS normalization: 1.0% of measured traffic on
//! Qwen3.5 and 1.7% on gemma-3-270m.
//!
//! ```text
//! sum_xx = Σ x²        N = extent(col)
//! rrms   = 1 / sqrt(sum_xx/N + eps)
//! dst    = x · rrms
//! ```
//!
//! This follows `ggml_compute_forward_rms_norm_f32` term for term. The native
//! kernel accumulates in `double` and then normalizes in F32; this deterministic
//! reduction accumulates in F32, as both native GPU kernels already do - the
//! benchmark's parity reference is the CPU, and the `test-backend-ops`
//! tolerance is also passed by the native Metal kernel.
//!
//! **What this kernel requires from nobody.** Nothing. It adds no vocabulary to
//! Loop IR: `AxisExtent` has existed since `rms_norm_back`, as have the
//! deterministic reduction and `SharedTree`. This is exactly why it is grouped
//! with `ACC` and `UNARY` - a new op over an already measured problem shape,
//! where the only thing to specify is the shape rule between its two lowerings.
//!
//! **The domain.** ggml requires `src0` and `dst` to have the same shape, and the
//! native Metal kernel gates the op on `ggml_is_contiguous_rows(src0)` and F32.
//! `supports_op` derives shape equality because both arguments are indexed by
//! the same axes; F32 is declared as a constraint.
//!
//! **What remains native, which must be stated before measuring.** Metal fuses
//! `RMS_NORM` with the following `MUL` and subsequent `ADD` - the
//! `ggml_metal_op_norm` pattern, saving two dispatches per layer. A RIR kernel
//! computes *one* node; dispatch therefore hands it control only when fusion
//! found nothing, exactly like the elementwise strip.
//! This is not a domain restriction - the fused node is not rejected, it is
//! never offered - and explains why served traffic is below measured traffic.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, ScalarType, TensorType,
    ValidateError, ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("rms_norm");

    let x = k.input("x", TensorType::f32(4));
    let dst = k.output("dst", TensorType::f32(4));
    let eps = k.param("eps", ScalarType::F32);

    // Same declaration order as `rms_norm_back`, with which this kernel shares
    // the schedule table: `row` on x, `plane` on y, `batch` on z; `col` remains
    // the inner reduction axis.
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let idx = [col, row, plane, batch];

    let xv = k.read(x, &idx);
    let xx = k.mul(xv, xv);
    let sum_xx = k.reduce(ReduceOp::Sum, col, xx, ReductionSemantics::Deterministic);

    let n = k.axis_extent(col);
    let mean = k.div(sum_xx, n);
    let mean_eps = k.add(mean, eps);
    let one = k.const_f32(1.0);
    let rms = k.sqrt(mean_eps);
    let rrms = k.div(one, rms);

    let value = k.mul(xv, rrms);
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

    /// Independent analytical reference, written from
    /// `ggml_compute_forward_rms_norm_f32`, not from the kernel.
    fn reference(
        x: &[f32],
        dst: &mut [f32],
        n_col: usize,
        n_row: usize,
        stride_elems: usize,
        eps: f32,
    ) {
        for r in 0..n_row {
            let base = r * stride_elems;
            let mut sum = 0.0f64;
            for c in 0..n_col {
                sum += (x[base + c] * x[base + c]) as f64;
            }
            let mean = (sum / n_col as f64) as f32;
            let scale = 1.0f32 / (mean + eps).sqrt();
            for c in 0..n_col {
                dst[base + c] = x[base + c] * scale;
            }
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

    fn run_case(schedule: Schedule, n_col: usize, n_row: usize, stride: usize, zero_x: bool) {
        let kernel = build().unwrap();
        let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
        let eps = 1e-5f32;

        let len = stride * n_row;
        let mut seed = 0x3d17_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
        let mut x = vec![0f32; len];
        if !zero_x {
            fill(&mut seed, &mut x, 1.5);
        }

        let mut expected = vec![0f32; len];
        reference(&x, &mut expected, n_col, n_row, stride, eps);

        let mut got = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::strided_2d(&x, n_col, n_row, stride)),
            BoundArg::Out(TensorViewMut::strided_2d(&mut got, n_col, n_row, stride)),
        ];
        run(&lk, &mut args, &[eps]).unwrap();

        for r in 0..n_row {
            let base = r * stride;
            for c in 0..n_col {
                let (g, e) = (got[base + c], expected[base + c]);
                let tol = 1e-4f32.max(2e-4 * e.abs());
                assert!(
                    (g - e).abs() <= tol,
                    "{:?} {n_col}x{n_row} stride {stride} [{r},{c}] : {g} vs {e}",
                    schedule.backend()
                );
            }
            for c in n_col..stride {
                assert_eq!(got[base + c], 0.0, "write outside logical elements");
            }
        }
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for &(n_col, n_row) in &[(1, 1), (3, 5), (33, 7), (64, 4), (1, 9), (17, 1)] {
            run_case(Schedule::cpu_serial(), n_col, n_row, n_col, false);
        }
    }

    #[test]
    fn parity_with_non_contiguous_strides() {
        for &(n_col, n_row, stride) in &[(3, 5, 8), (33, 7, 40), (1, 4, 3)] {
            run_case(Schedule::cpu_serial(), n_col, n_row, stride, false);
        }
    }

    /// `x = 0`: the mean is zero, so `dst` is `0 / sqrt(eps)`, hence zero. This
    /// is the branch protected by epsilon - without it the kernel would divide
    /// by zero and return NaNs where ggml returns zeros.
    #[test]
    fn the_epsilon_floor_when_the_row_is_zero() {
        run_case(Schedule::cpu_serial(), 9, 3, 9, true);
    }

    /// The two GPU lowerings executed by the fork: the 32-lane subgroup
    /// reduction, then the shared tree of 256 reserved by the shape rule for
    /// wide, sparse rows.
    #[test]
    fn parity_of_the_gpu_lowerings_against_the_reference() {
        for &(n_col, n_row, stride) in &[(1, 1, 1), (7, 3, 7), (33, 5, 40), (256, 2, 300)] {
            run_case(Schedule::vulkan_subgroup(), n_col, n_row, stride, false);
        }
        for &(n_col, n_row, stride) in &[(256, 2, 256), (1024, 4, 1024)] {
            run_case(
                Schedule::vulkan_shared_reduce(),
                n_col,
                n_row,
                stride,
                false,
            );
            run_case(Schedule::metal_shared_reduce(), n_col, n_row, stride, false);
        }
    }

    /// The shape class motivating the three outer axes: `x` occupies one plane
    /// out of three in a packed tensor, which no repetition can express.
    #[test]
    fn parity_on_a_rank_4_view_with_a_plane_gap() {
        let kernel = build().unwrap();
        for schedule in [Schedule::cpu_serial(), Schedule::vulkan_subgroup()] {
            let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
            let (n_col, n_row, n_plane, n_batch, gap) = (33usize, 4usize, 3usize, 2usize, 3usize);

            let packed = |g: usize| {
                [
                    4,
                    4 * n_col,
                    4 * n_col * n_row * g,
                    4 * n_col * n_row * g * n_plane,
                ]
            };
            let len = |g: usize| n_col * n_row * n_plane * n_batch * g;

            let mut seed = 0x9c4a_2026u64;
            let mut x = vec![0f32; len(gap)];
            fill(&mut seed, &mut x, 1.5);
            let eps = 1e-5f32;

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
            run(&lk, &mut args, &[eps]).unwrap();

            for b in 0..n_batch {
                for p in 0..n_plane {
                    for r in 0..n_row {
                        let dst_base = ((b * n_plane + p) * n_row + r) * n_col;
                        let x_base = ((b * n_plane * gap + p * gap) * n_row + r) * n_col;
                        let mut sum_xx = 0.0f32;
                        for c in 0..n_col {
                            sum_xx += x[x_base + c] * x[x_base + c];
                        }
                        let rrms = 1.0f32 / (sum_xx / n_col as f32 + eps).sqrt();
                        for c in 0..n_col {
                            let e = x[x_base + c] * rrms;
                            let g = got[dst_base + c];
                            let tol = 1e-4f32.max(2e-4 * e.abs());
                            assert!(
                                (g - e).abs() <= tol,
                                "{:?} [{b},{p},{r},{c}] : {g} vs {e}",
                                schedule.backend()
                            );
                        }
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/rms_norm/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (33usize, 7usize, 40usize);
        let len = stride * n_row;
        let mut seed = 0xc0de_2027u64;
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut x, 1.5);
        let eps = 1e-5f32;

        let mut expected = vec![0f32; len];
        reference(&x, &mut expected, n_col, n_row, stride, eps);

        let mut got = vec![0f32; len];
        let nb = [4usize, 4 * stride, 4 * stride * n_row, 4 * stride * n_row];
        generated::rms_norm(
            n_row,
            1,
            1,
            n_col,
            generated::TensorRef { data: &x, nb },
            generated::TensorRefMut { data: &mut got, nb },
            eps,
        );

        for r in 0..n_row {
            let base = r * stride;
            for c in 0..n_col {
                let (g, e) = (got[base + c], expected[base + c]);
                let tol = 1e-4f32.max(2e-4 * e.abs());
                assert!((g - e).abs() <= tol, "generated CPU [{r},{c}]: {g} vs {e}");
            }
        }
    }
}
